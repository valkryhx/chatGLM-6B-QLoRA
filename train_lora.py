# -*- coding: utf-8 -*-
# time: 2023/7/7 10:13
# file: train_qlora_deepspeed_zero.py
# author: hx
# email: XXXXX

import os
import argparse
from typing import List, Dict, Optional
from accelerate import init_empty_weights  # load an empty model,just structure , no real weight.
import bitsandbytes as bnb
import torch
from glob import glob
from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig 
)
from peft import (
    TaskType,
    LoraConfig,
    AdaLoraConfig ,  #  提出自2020年 感觉和lora区别不大 而且和qlora有冲突 这里代码没有用到 
                     #例子https://www.zhihu.com/question/596950521/answer/3109759716
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import json

#from modeling_chatglm import ChatGLMForConditionalGeneration
#from tokenization_chatglm import ChatGLMTokenizer
_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}


def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B QLoRA')
    parser.add_argument('--train_args_json', type=str, required=True, help='TrainingArguments的json文件')
    parser.add_argument('--model_name_or_path', type=str, default='THUDM/chatglm-6b', help='模型id或local path')
    parser.add_argument('--train_data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--eval_data_path', type=str, default=None, help='验证数据路径')
    parser.add_argument('--seed', type=int, default=42)
    #parser.add_argument('--max_input_length', type=int, default=256, help='instruction + input的最大长度')
    #parser.add_argument('--max_output_length', type=int, default=256, help='output的最大长度')
    parser.add_argument('--max_length', type=int, default=512, help='q+a的最大token个数 ，注意一个token可能对应2-3个char')
    parser.add_argument('--lora_rank', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora_alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--prompt_text', type=str, default='', help='统一添加在所有数据前的指令文本')
    parser.add_argument('--compute_dtype', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'], help='计算数据类型')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default="ds_zero2_config.json")
    parser.add_argument("--output_dir",type=str,default="output/lora",help="模型训练输出目录")
    parser.add_argument("--per_device_train_batch_size",type=int,default=1)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=1)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--learning_rate",type=float,default=2e-5)
    parser.add_argument("--num_train_epochs",type=float,default=1.0)
    parser.add_argument("--num_train_samples",type=int,default= 0,help="用于train的样本数量，可选。")
    parser.add_argument("--num_eval_samples",type=int,default= 0,help="用于eval的样本数量，可选。")
    parser.add_argument("--save_total_limit" , type=int ,default=None)
    #parser.add_argument("--load_in_4bit" , type=bool ,default=False)
    #parser.add_argument("--load_best_model_at_end",type=bool,default=True)  # https://huggingface.co/docs/transformers/main_classes/trainer
    parser.add_argument('--load_in_4bit',
                      help='Whether to load_in_4bit',
                      type=eval, 
                      choices=[True, False], 
                      default='False')
    parser.add_argument('--load_best_model_at_end',
                      help='Whether to load_best_model_at_end',
                      type=eval, 
                      choices=[True, False], 
                      default='True')
    parser.add_argument('--use_qlora',
                      help='Whether to use qlora',
                      type=eval, 
                      choices=[True, False], 
                      default='False')
    #"output_dir": "output/qlora_ds_zero",
    #"per_device_train_batch_size": 8, 
    #"per_device_eval_batch_size":  2,
    #"gradient_accumulation_steps": 8,
    #"learning_rate": 2e-5,
    #"num_train_epochs": 10.0,

    
    return parser.parse_args()


def tokenize_func(example, tokenizer, global_args, ignore_label_id=-100):
    """单样本tokenize处理
       q="你是谁？"
       a="你是谁"
       total=q+a       
       q_ids = tk.encode(text=q, add_special_tokens=False)
       a_ids = tk.encode(text=a, add_special_tokens=False)
       total_ids = tk.encode(text=total,add_special_tokens=False)
       可以看出      
       q_ids = [30910, 34607, 55622, 31514]
       a_ids = [30910, 34607, 55622]
       total_ids = [30910, 34607, 55622, 31514, 34607, 55622]
       total_ids 是拼接后的q+a 的ids 会省去一个a_ids的开头的30910 但是前半部分和q_ids一模一样
       我直接用 total_ids[:len(q_ids)]来表示q_ids没有问题，后面正好是a_ids[1:]，正好也不需要那个30910
       那么label也就是question_length = len(q_ids)
                      labels = [ignore_label_id] * question_length + total_ids[question_length:]
    """
    question = global_args.prompt_text + example['instruction']
    if example.get('input', None):
        if example['input'].strip():
            question += f'''\n{example['input']}'''
    answer = example['output']
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
    #这里不使用 tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)，因为这个方法可能其他模型没有
    input_ids = tokenizer.encode(text= question + answer ,add_special_tokens=False)[:global_args.max_length]
    #可能需要加上下面的
    #input_ids = input_ids + [2] # [2]是eos token id 确保最后是结束符eos 不然训练后可能模型会出现一直输出的现象。
    q_length = len(q_ids)
    labels = [ignore_label_id] * q_length + input_ids[q_length:]
    assert len(input_ids)  == len(labels) 
    return {'input_ids': input_ids, 'labels': labels}

#20230923 这个不再使用
def tokenize_func_old_not_use(example, tokenizer, global_args, ignore_label_id=-100):
    """单样本tokenize处理"""
    question = global_args.prompt_text + example['instruction']
    if example.get('input', None):
        if example['input'].strip():
            question += f'''\n{example['input']}'''
    answer = example['output']
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
    # tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)会在开头位置添加额外2个token:gmask, bos。所里这里如果q_ids超了 就让q_ids舍去最后2个token 
    if len(q_ids) > global_args.max_input_length - 2:  
        q_ids = q_ids[: global_args.max_input_length - 2]
    # tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)会在结束添加额外1个token:eos。所里这里如果a_ids超了 就让a_ids舍去最后1个token
    if len(a_ids) > global_args.max_output_length - 1:  # 额外token为1  eos
        a_ids = a_ids[: global_args.max_output_length - 1]
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    # question_length = input_ids.index(tokenizer.bos_token_id)
    # chatglm1 - gmask, bos,这个没用过先不管 #chatglm2 会在最开始的位置多两个token 64790, 64792 ,可能就是gmask, sop
    # 所以需要 +2 才是真正的q_ids的结束位置
    question_length = len(q_ids) + 2  
    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    
    return {'input_ids': input_ids, 'labels': labels}


def get_datset(data_path, tokenizer, global_args,max_samples=None):
    """读取本地包含json/jsonl文件的目录，将目录中所有文件作为dataset，并tokenize，shuffle，返回datasets.dataset"""
    
    if not (data_path is not None and os.path.exists(data_path)):
        raise ValueError("data_path requires a directory pointing to   jsons/jsonls")
    """https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py#L383"""
    data_files_list = glob(f'{data_path}/**/*.json', recursive=True) + glob(
                f'{data_path}/**/*.jsonl', recursive=True)
    logger.info(f"data files: {', '.join(data_files_list)}")
          
    data = load_dataset('json', data_files=data_files_list)
    ''' 只使用max_samples 个样本来参与train和eval  
        https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py#L453
    '''
    logger.info(f"在取样之前 data len ={len(data['train'])}")
    if max_samples is not None and max_samples > 0:
            max_samples = min(len(data['train']), max_samples)  # 
            data['train'] =  data['train'].select(range(max_samples))
    logger.info(f"在取样之后 data len ={len(data['train'])}")
    
    column_names = data['train'].column_names  # remove_columns=column_names  ,remove all at once
    """tokenize_func 中是单样本处理的写法 所以这里的batched只能设置为False"""
    logger.info("preprocessing dataset...")
    dataset = data['train'].map(lambda example: tokenize_func(example, tokenizer, global_args),
                                batched=False, 
                                remove_columns=column_names)
    dataset = dataset.shuffle(seed=global_args.seed)
    dataset = dataset.flatten_indices()
    return dataset

"""与这个一致 https://github.com/valkryhx/LLM-Tuning/blob/master/chatglm2_lora_tuning.py#L37
   只不过使用类方式实现 
"""
class DataCollatorForChatGLM:
    def __init__(self,
                 pad_token_id: int,
                 max_length: int = 2048,
                 ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """根据batch最大长度做padding"""
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[: self.max_length]
                label = label[: self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        #logger.error(f"len_of_input_ids={len(input_ids)}")
        #logger.error(f"len_of_labels={len(labels)}")
        return {'input_ids': input_ids, 'labels': labels}




class LoRATrainer(Trainer):
    print("into lora trainer !!!")
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        logger.error("Begin to save...")
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_world_process_zero():  
            self.model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            logger.error("Save done.")
        else :
            print("this process is not main process , do not save model.[for distributed training scenario]")

def find_all_linear_names(model,use_qlora=False):
    """
    找出所有全连接层，为所有全连接添加adapter
    实际返回的是个list，例如['dense', 'query_key_value', 'dense_4h_to_h', 'dense_h_to_4h']
    所以388行的loraConfig中target_modules=target_modules  实际要的参数是传入list ['query_key_value'] 而不是str 'query_key_value'
    
    """
    cls = torch.nn.Linear
    logger.debug(f"use_qlora={use_qlora},type={type(use_qlora)}")
    if use_qlora==True:
        cls = bnb.nn.Linear4bit
        logger.debug("use qlora in find all_linear_names")
    else:
        logger.debug("not use qlora in find all_linear_names")
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if  'output_layer' in lora_module_names:
        lora_module_names.remove('output_layer')
    logger.info(f"modules to add lora :\n{list(lora_module_names)}")
    #raise ValueError("debug here")
    return list(lora_module_names)


def train(global_args):


    # ADD by hx 20230629
    # https://huggingface.co/docs/transformers/main_classes/deepspeed 参考 【Constructing Massive Models】一节 
    # 直接在TrainingArgs中加入deepspeed=ds_config即可
    # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
    
    ''' ds_config 变量 可以用于在程序内导入 本次使用的是外部json文件'''
    ds_config ={
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": False
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
            "adam_w_mode": True
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        },
        "offload_param": {
            "device": "none",
            "pin_memory": False
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e6,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e6,
        "stage3_max_reuse_distance": 1e6,
        "stage3_gather_16bit_weights_on_model_save": True
    },    
    "train_batch_size": 8 ,  
    "train_micro_batch_size_per_gpu":4
}
    '''这两个参数train_batch_size 和 train_micro_batch_size_per_gpu好像在训练过程中没生效 
       训练实际的batchsize是chatGLM_6B_QLoA.json 中的参数决定的 '''
    
    

    
    hf_parser = HfArgumentParser(TrainingArguments)
    '''读取json中默认配置作为训练参数'''
    hf_train_args, = hf_parser.parse_json_file(json_file=global_args.train_args_json)

    
    with open(global_args.deepspeed,'r',encoding='utf-8') as fr:   # 这里就是向TrainingArgs中添加deepseed字段
        hf_train_args.deepspeed = json.load(fr)  # set trainingArgs中deepspeed=ds_config

    '''读取命令行传入参数 这个优先级高  覆盖对应的默认参数'''
    set_seed(global_args.seed)
    hf_train_args.seed = global_args.seed
    hf_train_args.optim="paged_adamw_8bit"

    hf_train_args.output_dir = global_args.output_dir 
    hf_train_args.logging_dir = global_args.output_dir 
    hf_train_args.per_device_train_batch_size = global_args.per_device_train_batch_size
    hf_train_args.per_device_eval_batch_size = global_args.per_device_eval_batch_size
    hf_train_args.gradient_accumulation_steps = global_args.gradient_accumulation_steps
    hf_train_args.learning_rate = global_args.learning_rate
    hf_train_args.num_train_epochs = global_args.num_train_epochs
    hf_train_args.save_total_limit = global_args.save_total_limit
    model_max_length = global_args.max_length 
    #model_max_length = global_args.max_input_length + global_args.max_output_length
    tokenizer = AutoTokenizer.from_pretrained(global_args.model_name_or_path, trust_remote_code=True)

    # Quantization
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=_compute_dtype_map[global_args.compute_dtype]) if global_args.use_qlora else None
    print(f"q_config={q_config}")
    
    
        
    model = AutoModel.from_pretrained(global_args.model_name_or_path,
                                          trust_remote_code=True,                           
                                          load_in_4bit=True if global_args.use_qlora else False ,   
                                          # 如果用qlora的话 这里写True/False都不影响 因为都会被q_config的load_in_4bit=True覆盖   
                                          # 通过True/False条件下的logger.error("加载完基座模型layers[27].self_attention.query_key_value.weight")确认layer是一样的unit8值
                                          torch_dtype=torch.float16,
                                          #quantization_config=q_config if global_args.use_qlora==True else None,
                                          quantization_config=q_config,
                                          empty_init=False,   # https://github.com/THUDM/ChatGLM-6B/issues/530
                                          #device_map=new_hf_device_map,
                                          # device_map="auto"   # add 20230713
                                     )
    # base_model 先设置use_cache=False 后面peftModel也会设置一次 在Trainer.train之前
    # 不使用old的past_query_key_value  设置为True的话是为了加快训练 但是用的是旧的qkv
    model.config.use_cache= False
    logger.debug(f"base model.config.use_cache={model.config.use_cache}")
    logger.error("加载完基座模型layers[27].self_attention.query_key_value.weight")
    print(model.transformer.encoder.layers[27].self_attention.query_key_value.weight)
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    # 
    # .gradient_checkpointing_enable()
    # .enable_input_require_grads()
    # .is_parallelizable
    # 这三个都是 transformers 模型的函数/参数(见 transformers/modeling_utils.py 文件)
    #
    
    model.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    model.enable_input_require_grads()
    # note: Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed. 
    # See https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/modeling_utils.py#L1190


    # STEP 4 : 将model转化为peftModel 准备loRA微调
    logger.info(f"global_args.use_qlora={global_args.use_qlora}")
    if global_args.use_qlora:
        logger.info("prepare_model_for_kbit_training...")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # LoRA
    #target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
    target_modules = find_all_linear_names(model,global_args.use_qlora)
    
    lora_config = LoraConfig(   # AdaLoraConfig 和 qlora好像有冲突 或者是多卡有冲突
        r=global_args.lora_rank,
        lora_alpha=global_args.lora_alpha,
        target_modules=target_modules, #["query_key_value"],只对qkv而不对dense层加lora的话 效果不好 
        lora_dropout=global_args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    logger.error("加上lora层后layers[27].self_attention.query_key_value.weight")
    print(model.base_model.model.transformer.encoder.layers[27].self_attention.query_key_value.weight)
    resume_from_checkpoint = global_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')

    model.print_trainable_parameters()

    # data
    logger.info("loading dataset...")
    train_dataset = get_datset(global_args.train_data_path, tokenizer, 
                               global_args,
                               max_samples=global_args.num_train_samples )
    
    """ 
     eval data数据量太少（比如4）会而且 gradiant accumulationc较大时（比如8）和batchsize , num_gpu较大时无法计算和积累梯度
     eval_data至少要 >= 后面3者的乘积
     RuntimeError: unscale_() has already been called on this optimizer since the last update().
     https://github.com/huggingface/transformers/issues/23935#issuecomment-1597170127
     """
    eval_dataset = None
    if global_args.eval_data_path:
        eval_dataset = get_datset(global_args.eval_data_path, tokenizer, 
                                  global_args,
                                  max_samples=global_args.num_eval_samples)

    data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id,
                                           max_length=model_max_length)

    # print hf_train_args to see the manually set paras
    print(f"number_train_samples={len(train_dataset)}\nnumber_of_eval_numbers={len(eval_dataset)}")
    print(hf_train_args)
    # raise ValueError("TEST")
    
    # train
    logger.debug(f'hf_train_args.per_device_train_batch_size={hf_train_args.per_device_train_batch_size}')
    logger.debug(f'hf_train_args.per_device_eval_batch_size ={hf_train_args.per_device_eval_batch_size }')
    logger.debug(f'hf_train_args.gradient_accumulation_steps ={hf_train_args.gradient_accumulation_steps }')
    logger.debug(f"在刚刚加载完base_model就写成False 那下面的设置就不用写了 会也是False 否则在下面设置也行\nmodel.config.use_cache={model.config.use_cache}")
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    logger.debug(f"现在写就是peft_model 效果跟base_model一样 model.config.use_cache={model.config.use_cache}")
    
    trainer = LoRATrainer(
        model=model,
        args=hf_train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(hf_train_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
