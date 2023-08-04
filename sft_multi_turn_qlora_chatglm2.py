# -*- coding: utf-8 -*-
# time: 2023/8/2 14:27
# file: pretrain_qlora_chatglm2.py
# author: hx
# https://github.com/valkryhx


"""
使用方式：注意在代码中已经根据chatglm的多轮对话代码加入了round_i_问_答格式  所以数据集中就保持原样的history就行，
{"history": [["我今天想去玩术士。", "你要打toc吗？"], ["是的，我要买饰品。", "老四的裁决？"], ["怎么可能？我要买小强的亡者君临", "哦哦对 裁决是物理毕业 不是SS的"]]}
代码中的处理方式参考【https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenization_chatglm.py#L167】

!git pull --all --force 
#!pip install  -U git+https://github.com/huggingface/peft.git   # 20230717 peft==0.4.0正式发布了 不用调版本了推理完后再训练需要重新升级到0.4.0dev 所以有这句
!deepspeed --include localhost:0,1  sft_qlora_chatglm2.py  \
  --train_args_json luzi.json \
  --model_name_or_path THUDM/chatglm2-6b \
  --output_dir output-multi-turn-sft-0802-v1 \
  --num_train_samples -1 \
  --num_eval_samples 5 \
  --train_data_path ./data/multi_turn_conversations   \
  --eval_data_path  ./data/multi_turn_conversations     \
  --max_length 1024 \
  --lora_rank 64 \
  --lora_dropout 0.05 \
  --compute_dtype fp16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --gradient_accumulation_steps 1 \
  --learning_rate  5e-5 \
  --num_train_epochs  40  \
  --save_total_limit 2 \
  --load_in_4bit True \
--deepspeed ds_zero2_config.json

"""



import random
import os
import argparse
from typing import List, Dict, Optional, Any, Mapping
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
    #AdaLoraConfig ,  #  提出自2020年 感觉和lora区别不大 而且和qlora有冲突 这里代码没有用到 
                     #例子https://www.zhihu.com/question/596950521/answer/3109759716
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import json
from itertools import chain

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}


# 关闭dataset的cache 这样每次都重新生成 测试用 不用cache避免使用到旧数据集
from datasets import set_caching_enabled
set_caching_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B QLoRA')
    parser.add_argument('--train_args_json', type=str, required=True, help='TrainingArguments的json文件')
    parser.add_argument('--model_name_or_path', type=str, default='THUDM/chatglm-6b', help='模型id或local path')
    parser.add_argument('--train_data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--eval_data_path', type=str, default=None, help='验证数据路径')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_input_length', type=int, default=512, help='多个轮次对话的总文本的最大长度 也就是history对应的多个对话的Q+A的整体长度')
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
    parser.add_argument("--num_train_samples",type=int,default= -1,help="用于train的样本数量，可选,若为-1则是全部样本。")
    parser.add_argument("--num_eval_samples",type=int,default= -1,help="用于eval的样本数量，可选，若为-1则是全部样本。")
    parser.add_argument("--save_total_limit" , type=int ,default=None)
    parser.add_argument("--load_in_4bit" , type=bool ,default=True)
    parser.add_argument("--load_best_model_at_end",type=bool,default=True)  # https://huggingface.co/docs/transformers/main_classes/trainer
    parser.add_argument("--block_size",type=int,default=256,help="将篇章级别文本分词后的长tokens结果 按照block_size划分成固定大小 想象一下长火车分成多个车厢")
    parser.add_argument("--max_length",type=int,default=256,help="每个样本的最大长度，一般会小于等于block_size")
    parser.add_argument("--data_type",type=str,default="history",choices=['history', 'sharegpt'],help="每个样本的最大长度，一般会小于等于block_size")
    #"output_dir": "output/qlora_ds_zero",
    #"per_device_train_batch_size": 8, 
    #"per_device_eval_batch_size":  2,
    #"gradient_accumulation_steps": 8,
    #"learning_rate": 2e-5,
    #"num_train_epochs": 10.0,  
    return parser.parse_args()


## 20230731 根据shiming64的代码和 firefly的代码 理解了decoder 模型输入input_ids 都是完整的q+a，注意是q+a而不是q ,用labels来标记正确答案
## 也就是最终的trainer中额dataset 无论是train / eval 都是字典形式即可
## 即 {"input_ids":list[多个input_ids]
##       "labels":list[多个labels]}
## pt阶段 labels = input_ids.copy()也就是说labels中每个token都要计算loss
## sft阶段 把labels中不需要计算loss的部分掩盖住，对于单轮qa 就是 
# input_ids=Q+A 
# labels= mask_q_len *[ignore_token] + A 
# 看到没 input_ids和 labels长度和内容本来是一致的 只不过labels将其中不用计算loss的内容做了替换
## 多轮对话也是可以这么处理 根据shiming64和firefly对于百川的处理 多轮对话的input_ids=Q1+A1+Q2+A2 +... 把所有历史对话当前对话全部拼起来
## labels长度也是input_ids长度 只不过需要仔细拼接 labels =第一段Q1的长度mask+ A1 + 第二段Q2长度mask + A2 + ...
## 这样的多轮对话处理方式相当于并行计算所有对话的loss 提升了样本利用率（不仅仅是只根据大段历史对话来计算最后一轮对话）
## llama好像默认就这么处理的 


# 用于普通history格式数据集的处理
def tokenize_function_history(example,tokenizer,ignore_label_id: int = -100, max_length=8192):  # 在get_multi_turn_conversations_datset函数中用到了
    """
       多轮对话使用每个example（也就是一行json样本）中的example['history'] 拼接多轮对话 构建一个包含多轮对话的总的input_ids和总的labels
       这个Q_temp和A_temp 不同的model 都不一样 但是很重要 
       这里用chatglm2-6b官网的tokenization_chatglm.py中的
    """
    Q_temp = "[Round {}]\n\n问：{}"
    A_temp = "\n\n答：{}\n\n"

    ## modify 20230803
    #Q_temp = "问：{}"
    #A_temp = "答：{}"
    input_ids =[]
    labels =[]
    for turn_number,conversation in enumerate(example['history']):
        #print(turn_number + 1)
        q = conversation[0]
        a = conversation[1]
        #print(Q_temp.format(turn_number+1,q))
        #print(A_temp.format(a))
        Q = Q_temp.format(turn_number+1,q)  ## modify 20230803
        #Q = Q_temp.format(q)   ## ## modify 20230803
        A = A_temp.format(a)
        Q_token_list = tokenizer.encode(Q,add_special_tokens=False) 
        #print(f"在每轮Q-A后补充一个 tokenizer.eos_token_id = {tokenizer.eos_token_id}")
        A_token_list = tokenizer.encode(A,add_special_tokens=False) + [tokenizer.eos_token_id] # 在每一个Q-A对话的[A ]后面加[</s> id]
        input_ids.extend(Q_token_list)
        input_ids.extend(A_token_list)
        labels.extend([ignore_label_id]*len(Q_token_list))
        labels.extend(A_token_list)
    #print(f"input_ids={input_ids}")
    #print(f"labels={labels}")
    assert len(input_ids) == len(labels) # 这里一定要保证长度相等 不然说明拼接出问题
    #input_ids_token_to_str = tokenizer.batch_decode(input_ids,skip_special_tokens=True)
    #labels_token_to_str = tokenizer.batch_decode(labels,skip_special_tokens=True)
    #print(input_ids_token_to_str)
    #print(labels_token_to_str)
    
    # cut and padding
    input_ids = input_ids[:max_length]
    labels  = labels[:max_length]
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    labels  = labels + [ignore_label_id ] * pad_len
    return {
            "input_ids":input_ids , 
            "labels": labels,
            "attention_mask" :torch.Tensor(input_ids).ne(tokenizer.pad_token_id).int(),   # input_ids 中为pad token attenmask =0 其余token attenmask =1
            ## https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py#L754 
            ## https://cloud.tencent.com/developer/article/1885829
            }

# 用于shareGPT 格式数据集的处理
def tokenize_function_sharegpt(example,tokenizer,ignore_label_id = -100 ,max_length=8192): # 在get_multi_turn_conversations_datset函数中用到了
    """
       多轮对话使用每个example（也就是一行json样本）中的example['history'] 拼接多轮对话 构建一个包含多轮对话的总的input_ids和总的labels
       这个Q_temp和A_temp 不同的model 都不一样 但是很重要 
       这里用chatglm2-6b官网的tokenization_chatglm.py中的
    """
    Q_temp = "[Round {}]\n\n问：{}"
    A_temp = "\n\n答：{}\n\n"
    
    #Q_temp = "问：{}"
    #A_temp = "答：{}"
    
    input_ids =[]
    labels =[]
    must_be_even_len = len(example['conversations']) //2 *2
    if must_be_even_len == 0:
        print(f"must_be_even_len=0!")
        print(f"长度不足的example={example}")
        raise ValueError("example出现了长度不足2的异常。")
    #assert must_be_even_len > 0
    # must_be_even_len > 0 这个 条件由filter保证
    # my_dataset['train'].filter(lambda example : len(example["conversations"])>1)
    
    for idx in range(0,must_be_even_len,2):
        #print(turn_number + 1)
        q = example['conversations'][idx]['value']
        a = example['conversations'][idx+1]['value']
        #print(Q_temp.format(turn_number+1,q))
        #print(A_temp.format(a))
        Q = Q_temp.format(int(idx//2)+1,q)
        A = A_temp.format(a)
        Q_token_list = tokenizer.encode(Q,add_special_tokens=False)
        #print(f"在每轮Q-A后补充一个 tokenizer.eos_token_id = {tokenizer.eos_token_id}")
        A_token_list = tokenizer.encode(A,add_special_tokens=False) + [tokenizer.eos_token_id] # 在每轮Q-A后面补充一个eos_token_id
        input_ids.extend(Q_token_list)
        input_ids.extend(A_token_list)
        labels.extend([ignore_label_id]*len(Q_token_list))
        labels.extend(A_token_list)
    #print(f"input_ids={input_ids}")
    #print(f"labels={labels}")
    assert len(input_ids) == len(labels) # 这里一定要保证长度相等 不然说明拼接出问题
    #input_ids_token_to_str = tokenizer.batch_decode(input_ids,skip_special_tokens=True)
    #labels_token_to_str = tokenizer.batch_decode(labels,skip_special_tokens=True)
    #print(input_ids_token_to_str)
    #print(labels_token_to_str)

    # 1.cut and 2.padding
    input_ids = input_ids[:max_length]
    #labels  = labels[:max_length]
    #pad_len = max_length - len(input_ids)
    #input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    #labels  = labels + [ignore_label_id ] * pad_len
    return {
            "input_ids":input_ids , 
            "labels": labels,
            #"attention_mask" :torch.Tensor(input_ids).ne(tokenizer.pad_token_id).int(), 
            ## https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py#L754 
            ## https://cloud.tencent.com/developer/article/1885829
            }

def get_multi_turn_conversations_datset(data_path, tokenizer, max_samples=-1,global_args=None):
    """读取本地包含json/jsonl文件的目录，将目录中所有文件作为dataset，只采样max_samples个参与训练/评估。
    并tokenize，shuffle，返回datasets.dataset
    DataCollatorForChatGLM做了padding和截断 不是在这里做的
    """
    
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
    if global_args.data_type.strip().lower() == "history":
        least_sample_number = 1   #  =1 表示 一行的sample json中 "history":[(Q,A)] 至少有一个(Q,A) 这也只是初步检查 总比不检查好
        customized_tokenize_function = tokenize_function_history
    elif global_args.data_type.strip().lower() == "sharegpt":
        least_sample_number = 2   #  =2 表示 一行的sample json中 "conversations":[{Q},{A}] 至少有2个{Q} {A}  这也只是初步检查 总比不检查好
        customized_tokenize_function = tokenize_function_sharegpt
    else :
        raise ValueError("数据集类型错误!")
    #list(example.keys())[0] 就是每行sample json dict最外层的唯一的那个key list(example.keys())只有一个元素，要么是history 要么是conversations 其他值也行 总之只有一个
    tokenized_dataset = data['train'].filter(lambda example : len(example[list(example.keys())[0]]) >= least_sample_number).map(
                                lambda example: customized_tokenize_function(example, tokenizer=tokenizer,max_length=global_args.max_length),
                                batched=False, 
                                remove_columns=data['train'].column_names)
    # 验证打印一些信息
    print(f"tokenized_dataset={tokenized_dataset}")
    print(f"tokenizer.decode(tokenized_dataset[0]['input_ids'],skip_special_tokens=False)=\n{tokenizer.decode(tokenized_dataset[0]['input_ids'],skip_special_tokens=False)}")
    print(f"tokenizer.decode(tokenized_dataset[0]['labels'],skip_special_tokens=False)=\n{tokenizer.decode(tokenized_dataset[0]['labels'],skip_special_tokens=False)}")
    print(f"tokenized_dataset[0]['input_ids']=\n{tokenized_dataset[0]['input_ids']}")
    print(f"tokenized_dataset[0]['labels']=\n{tokenized_dataset[0]['labels']}")
    #print(f"tokenized_dataset[0]['attention_mask']=\n{tokenized_dataset[0]['attention_mask']}")
    print(f"len(tokenized_dataset[0]['input_ids']={len(tokenized_dataset[0]['input_ids'])}")
    print(f"len(tokenized_dataset[0]['labels']={len(tokenized_dataset[0]['labels'])}")
    ## 验证完毕
    tokenized_dataset = tokenized_dataset.shuffle(seed = 42)
    tokenized_dataset = tokenized_dataset.flatten_indices()
    return tokenized_dataset








class DataCollatorForChatGLM:
    """ 注意这个类主要作用就是padding和截断
       输入是list[{input_ids:xxx,label(可选):YYY},{input_ids:xxx,label(可选):YYY},{input_ids:xxx,label(可选):YYY}]
       输出是dict {inpu_ids :[xxx,xxx,xxx,] ,lebals :[yyy,yyy,yyy]}
       List[Dict[str, List]]) -> Dict[str, torch.Tensor]
       # 用法
       #data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id,max_length=8192)
    """
    def __init__(self,
                 pad_token_id: int,
                 max_length: int = 2048,
                 ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """根据batch最大长度做padding
           注意 input_ids 已经是所有轮次对话Q+A+Q+A的拼接了 而且长度等于labels
           将list_of_{dict(input_ids,labels)} -> dict_of(input_ids:list  ,labels:list)
           所以叫做collator 收集 聚集
        """
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, labels ,attention_mask= [], [] ,[]
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            #if batch_max_len > self.max_length:
            right_len = min(batch_max_len,self.max_length)
            ids = d['input_ids'][: right_len]
            label = d['labels'][: right_len]
            #print(f"batch_max_len={batch_max_len}")
            
            pad_len = right_len - len(ids)
            
            ids = ids + [self.pad_token_id] * pad_len
            label = label + [self.ignore_label_id] * pad_len  # 注意这里是在右边padding 而且labels使用的是ignore_token_id

            # 查看用的 能发现每一batch的maxlen都不同 现在一个batch里面2个sample 经常一个padlen=0 正常
            #print(f"pad_len={pad_len}")
            #print(f"len_ids={len(ids)}")
            #print(f"len_label={len(label)}")
            #print(f"len_attmask={len(ids)}")
            #print("======")
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
            attention_mask.append(torch.Tensor(ids).ne(self.pad_token_id).int())
            
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        return {'input_ids': input_ids, 'labels': labels , 'attention_mask':attention_mask}

"""
# 验证get_dataset(包含了tokenizer_function)
ds = get_datset(data_path="/kaggle/working/multi_turn", tokenizer=tokenizer, max_samples=-1,global_args=None)
for item in ds:
    #print(item)
    print(tokenizer.decode(item['input_ids'],skip_special_tokens=False))
    print("======================")
# 验证 DataCollatorForChatGLM
very_clear_data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id,max_length=900)
very_clear_data_collator(ds)

"""




# 可以作为预训练的data collator  但是写的不怎么看得懂 只能看出来就是专门处理了key=label 其他部分没动
def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))
    #print("打印fault_tolerance_data_collator处理后的信息")
    #print(f"batch.keys()={batch.keys()},len_batch_key_labels={len(batch['labels'])}    ,len_batch_key_k={len(batch[k])}")
    return batch



class LoRATrainer(Trainer):
    print("into lora trainer !!!")
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        print("begin to save  !!!")
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_world_process_zero():  
            self.model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            print("save done !!!")
        else :
            print("this process is not main process , do not save model.[for distributed training scenario]")
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if  'output_layer' in lora_module_names:
        lora_module_names.remove('output_layer')
    return list(lora_module_names)


def train(global_args):

    ##STEP 0 从命令行获取参数，包括trainingArgs在内的，以及各类附属参数
    logger.info("loading parameters...")
    # ADD by hx 20230629
    # https://huggingface.co/docs/transformers/main_classes/deepspeed 参考 【Constructing Massive Models】一节 
    # 直接在TrainingArgs中加入deepspeed=ds_config即可
    # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
    
    hf_parser = HfArgumentParser(TrainingArguments)
    '''读取json中默认配置作为训练参数'''
    """
    注意下面代码的 , 很重要 ，根据https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py#L379C32-L379C32
     HfArgumentParser.parse_json_file 返回的是一个tuple 【outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        return tuple(outputs)】
        所以如果不加, 那hf_train_args就是一个tuple 而不是一个TrainingArguments对象 下面的类似hf_train_args.seed = global_args.seed 代码就会报错
        加上，逗号之后 那就把tuple解开了 
        （即使tuple里面只有一个TrainingArguments对象 也需要解开真操蛋！多个A,B=HfArgumentParser.parse_json_file  可能还看不出来，
        但是单个A=HfArgumentParser.parse_json_file  不加逗号就是错误的）
        那就拿到了真正的TrainingArguments对象的hf_train_args 真的操蛋
        那个源码中其他parse方法都是return tuple 都要注意
        例如 parse_yaml_file /  parse_dict /parse_args_into_dataclasses
    """
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
    
    
    tokenizer = AutoTokenizer.from_pretrained(global_args.model_name_or_path, trust_remote_code=True)


    ###STEP 1 加载train / eval data
   
    logger.info("loading dataset...")
    # train_dataset = get_datset(global_args.train_data_path, tokenizer, 
    #                            global_args,
    #                            max_samples=global_args.num_train_samples )
    
    # train_dataset = get_multi_turn_conversations(data_path=global_args.train_data_path,
    #                               tokenizer = tokenizer,
    #                               block_size = global_args.block_size,
    #                               global_args_max_length=global_args.max_length,
    #                               max_samples=global_args.num_train_samples)

    train_dataset =  get_multi_turn_conversations_datset(data_path=global_args.train_data_path, 
                                                         tokenizer=tokenizer, 
                                                         max_samples=global_args.num_train_samples,global_args=global_args)
    
    """ 
     eval data数据量太少（比如4）会而且 gradiant accumulationc较大时（比如8）和batchsize , num_gpu较大时无法计算和积累梯度
     eval_data至少要 >= 后面3者的乘积
     RuntimeError: unscale_() has already been called on this optimizer since the last update().
     https://github.com/huggingface/transformers/issues/23935#issuecomment-1597170127
     """
    eval_dataset = None
    # if global_args.eval_data_path:
    #     eval_dataset = get_datset(global_args.eval_data_path, tokenizer, 
    #                               global_args,
    #                               max_samples=global_args.num_eval_samples)
    
    # eval_dataset = get_dataset_for_pretrain(data_path=global_args.eval_data_path,
    #                               tokenizer = tokenizer,
    #                               block_size = global_args.block_size,
    #                               global_args_max_length=global_args.max_length,
    #                               max_samples=global_args.num_eval_samples)

    eval_dataset =  get_multi_turn_conversations_datset(data_path=global_args.eval_data_path, 
                                                         tokenizer=tokenizer, 
                                                         max_samples=global_args.num_eval_samples,global_args=global_args)
    
    ### STEP 2 定义 data collator
    very_clear_data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id,max_length=global_args.max_length)
     


    ### STEP 3  load model
    # Quantization
    q_config = BitsAndBytesConfig(load_in_4bit=False,#True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=_compute_dtype_map[global_args.compute_dtype])
    
    """
    if torch.cuda.device_count() > 1:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = (world_size != 1) # True(distributed training) or False(single gpu )
        global_args.ddp_find_unused_parameters = False if ddp else None
        
        model.hf_device_map['transformer.output_layer'] = model.hf_device_map['transformer.embedding']
        new_hf_device_map = model.hf_device_map
        model.cpu()
        del model
        torch.cuda.empty_cache()
        print(f'memory_allocated {torch.cuda.memory_allocated()}')
        print('loading real model...')

    
        global_args.ddp_find_unused_parameters = False
        model = AutoModel.from_pretrained(global_args.model_name_or_path,
                                          trust_remote_code=True,                           
                                          load_in_4bit=True,
                                          torch_dtype=torch.float16,
                                          quantization_config=q_config,
                                          device_map=new_hf_device_map)
        print("[real]",model.hf_device_map)
        """
        
    model = AutoModel.from_pretrained(global_args.model_name_or_path,
                                          trust_remote_code=True,                           
                                          load_in_4bit=global_args.load_in_4bit,
                                          torch_dtype=torch.float16,
                                          quantization_config=q_config,
                                           # empty_init这是最关键的参数 如果不设置 那即使用deepspeed也oom
                                  # 当您使用 AutoModel.from_pretrained() 方法加载预训练模型时，模型权重会被存储在 PyTorch 的 nn.Parameter 对象中。
                                  # 在没有指定 empty_init=False 参数时，nn.Parameter 对象的值将被初始化为全零的张量。
                                  # 但是，由于 nn.Parameter 对象不是真正的张量，而是具有元数据的张量包装器，因此无法将这些对象直接复制到 DeepSpeed 使用的元数据张量中。
                                  # 在指定 empty_init=False 参数后，nn.Parameter 对象将被初始化为包含预训练权重的张量，
                                  # 这使得 DeepSpeed 能够正常地将权重复制到元数据张量中
                                  # THUDM/chatglm2 估计modeling_chatglm.py 默认是True  好坑！
                                  # 果然 一查真的是 https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L732
                                          empty_init=False,   # https://github.com/THUDM/ChatGLM-6B/issues/530
                                          #device_map=new_hf_device_map,
                                          # device_map="auto"   # add 20230713
                                     )

    
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
    logger.info("prepare_model_for_kbit_training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # LoRA
    #target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(   # AdaLoraConfig 和 qlora好像有冲突 或者是多卡有冲突
        r=global_args.lora_rank,
        lora_alpha=global_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=global_args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

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

    

    # print hf_train_args to see the manually set paras
    print(f"number_train_samples={len(train_dataset)}\nnumber_of_eval_numbers={len(eval_dataset)}")
    print(hf_train_args)
    # raise ValueError("TEST")
    
    ### STEP  5   train
    trainer = LoRATrainer(
        model=model,
        args=hf_train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = very_clear_data_collator , #modify 不使用这个collator 试试 20230804
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(hf_train_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
