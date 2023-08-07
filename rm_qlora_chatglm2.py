# -*- coding: utf-8 -*-
# time: 2023/8/7 14:20
# file: rm_qlora_chatglm2.py
# author: hx
# https://github.com/valkryhx

import random
import os
import evaluate
import argparse
from typing import List, Dict, Optional, Any, Mapping, Union
from accelerate import init_empty_weights  # load an empty model,just structure , no real weight.
import bitsandbytes as bnb
import torch
import torch.nn as nn
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
    BitsAndBytesConfig ,
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    LlamaForSequenceClassification, 
    LlamaConfig, 
    LlamaTokenizer,
    AutoModelForSeq2SeqLM
)
from transformers.utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

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

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

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
    parser.add_argument('--ddp_find_unused_parameters',
                      help='This is a boolean flag.',
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


"""
   implement Reward Model for SeqClassification because GLM not support SeqCLS task in transformers
"""

##################################################################################
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss

##################################################################################
## part of codes are from https://github.com/valkryhx/ChatGLM-LoRA-RLHF-PyTorch/blob/4d3b8df2d6b7908a924b91b339f4468ed357761e/reward_model.py#L54

# 定义奖励模型 原理是在chatglm2模型的基础上加上v_head层 v_head层的shape为(hidden_size, 1) 也就是输出为一个float 值，注意dtype=torch.float32
class RewardModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, model, tokenizer):
        super().__init__(config)
        self.model_type = config.model_type
        print("model_type: ", self.model_type)
        self.pad_id = tokenizer.pad_token_id
        self.transformer = model
        #定义v_head时也要注意dtype是float32避免 RuntimeError: expected scalar type Float but found Half
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False, dtype=torch.float32) 
        self.loss_fn = PairWiseLoss()

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing_disable()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PreTrainedModel):
            module.gradient_checkpointing = value

    def reward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None
    ):
        batch_size = input_ids.shape[0]
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.model_type == "glm":
            hidden_states = transformer_outputs.mems[-1]

        elif self.model_type == "chatglm":
            hidden_states = transformer_outputs[0]
            seq_len, batch_size, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        elif self.model_type == "pangu":
            hidden_states = transformer_outputs[0]
            hidden_states = hidden_states.squeeze(1)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        assert len(hidden_states.shape) == 3

        # print("hidden_states: ", type(hidden_states), hidden_states.dtype)
        rewards = self.v_head(hidden_states).squeeze(-1)
        rewards = rewards.to(torch.float32)  # 定义v_head时也要注意dtype是float32避免 RuntimeError: expected scalar type Float but found Half
        rewards = rewards.mean(dim=-1)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)    # ensure shape is (B)

        assert len(rewards.shape) == 1 and rewards.shape[0] == batch_size

        return rewards

    def forward(
            self,
            chosen_input_ids=None,
            chosen_attention_mask=None,
            chosen_position_ids=None,
            rejected_input_ids=None,
            rejected_attention_mask=None,
            rejected_position_ids=None,
            past_key_values=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        if chosen_input_ids is not None:
            chosen_reward = self.reward(chosen_input_ids, attention_mask=chosen_attention_mask, position_ids=chosen_position_ids)
            # print("chosen_reward: ", chosen_reward.shape)
        else:
            chosen_reward = None

        if rejected_input_ids is not None:
            reject_reward = self.reward(rejected_input_ids, attention_mask=rejected_attention_mask, position_ids=rejected_position_ids)
            # print("reject_reward: ", reject_reward.shape)
        else:
            reject_reward = None

        if chosen_reward is not None and reject_reward is not None:
            loss = self.loss_fn(chosen_reward, reject_reward)   
        else:
            loss = None

        return {
            "loss": loss,
            "chosen_reward": torch.sigmoid(chosen_reward) if chosen_reward is not None else chosen_reward,
            "reject_reward": torch.sigmoid(reject_reward) if reject_reward is not None else reject_reward,
        }


def preprocess_function(example,tokenizer):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    # for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
    for question, response_j, response_k in zip(example["user_input"], example["completion_a"], example["completion_b"]):
        chatglm2_prompt = "[Round 1]\n\n问：{}\n\n答：{}\n\n"
        tokenized_j = tokenizer(
            #"Question: " + question + "\n\nAnswer: " + response_j, truncation=True
              chatglm2_prompt.format(question,response_j),truncation=True
              )
        tokenized_k = tokenizer(
            #"Question: " + question + "\n\nAnswer: " + response_k, truncation=True
              chatglm2_prompt.format(question,response_k),truncation=True
              )
        
        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
        

    return new_examples

def get_rm_datset(data_path, tokenizer, max_samples=-1,global_args=None):
    """读取本地包含json/jsonl文件的目录，将目录中所有文件作为dataset，只采样max_samples个参与训练/评估。
    并tokenize，shuffle，返回datasets.dataset
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
    
    tokenized_dataset = data['train'].map(
                                lambda example: preprocess_function(example, tokenizer=tokenizer),
                                batched=True, 
                                remove_columns=data['train'].column_names)
    # 验证打印一些信息
    # print(f"tokenized_dataset={tokenized_dataset}")
    # print(f"tokenizer.decode(tokenized_dataset[0]['input_ids'],skip_special_tokens=False)=\n{tokenizer.decode(tokenized_dataset[0]['input_ids'],skip_special_tokens=False)}")
    # print(f"tokenizer.decode(tokenized_dataset[0]['labels'],skip_special_tokens=False)=\n{tokenizer.decode(tokenized_dataset[0]['labels'],skip_special_tokens=False)}")
    # print(f"tokenized_dataset[0]['input_ids']=\n{tokenized_dataset[0]['input_ids']}")
    # print(f"tokenized_dataset[0]['labels']=\n{tokenized_dataset[0]['labels']}")
    # #print(f"tokenized_dataset[0]['attention_mask']=\n{tokenized_dataset[0]['attention_mask']}")
    # print(f"len(tokenized_dataset[0]['input_ids']={len(tokenized_dataset[0]['input_ids'])}")
    # print(f"len(tokenized_dataset[0]['labels']={len(tokenized_dataset[0]['labels'])}")
    
    return tokenized_dataset

# 不加datclass那就写__init__函数 参考https://github.com/valkryhx/chatGLM-6B-QLoRA/blob/main/sft_multi_turn_qlora_chatglm2.py#L325
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors: str = "pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch

class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        # print('inputs["input_ids_j"]: ', inputs["input_ids_j"].shape)
        # print('inputs["attention_mask_j"]: ', inputs["attention_mask_j"].shape)
        #rewards_j = model(chosen_input_ids=inputs["input_ids_j"], chosen_attention_mask=inputs["attention_mask_j"])["chosen_reward"]
        # print("rewards_j: ", type(rewards_j), rewards_j.shape)

        # print('inputs["input_ids_k"]: ', inputs["input_ids_k"].shape)
        # print('inputs["attention_mask_k"]: ', inputs["attention_mask_k"].shape)
        #rewards_k = model(rejected_input_ids=inputs["input_ids_k"], rejected_attention_mask=inputs["attention_mask_k"])["reject_reward"]
        # print("rewards_k: ", type(rewards_k), rewards_k.shape)

        total_rewards_j_k =  model(
                   chosen_input_ids=inputs["input_ids_j"]+inputs["input_ids_k"], 
                   chosen_attention_mask=inputs["attention_mask_j"] + inputs["attention_mask_k"])
        reward_j = total_rewards_j_k[:len(inputs["input_ids_j"])]["chosen_reward"]
        reward_k = total_rewards_j_k[len(inputs["input_ids_j"]):]["reject_reward"]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
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

# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)
  
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
    hf_train_args.ddp_find_unused_parameters = global_args.ddp_find_unused_parameters
    
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

    # train_dataset =  get_multi_turn_conversations_datset(data_path=global_args.train_data_path, 
    #                                                      tokenizer=tokenizer, 
    #                                                      max_samples=global_args.num_train_samples,global_args=global_args)

    train_dataset = get_rm_datset(data_path=global_args.train_data_path, tokenizer=tokenizer, max_samples=global_args.num_train_samples,global_args=global_args)
    
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

    # eval_dataset =  get_multi_turn_conversations_datset(data_path=global_args.eval_data_path, 
    #                                                      tokenizer=tokenizer, 
    #                                                      max_samples=global_args.num_eval_samples,global_args=global_args)
    eval_dataset = get_rm_datset(data_path=global_args.eval_data_path, tokenizer=tokenizer, max_samples=global_args.num_eval_samples,global_args=global_args)
    
    ### STEP 2 定义 data collator
    rm_data_collator = RewardDataCollatorWithPadding(
                                   tokenizer=tokenizer, 
                                   max_length=global_args.max_length, 
                                   pad_to_multiple_of=8)

  
     


    ### STEP 3  load model
    # Quantization
    q_config = BitsAndBytesConfig(load_in_4bit= True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=_compute_dtype_map[global_args.compute_dtype])
        
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
        task_type=TaskType.SEQ_CLS #TaskType.CAUSAL_LM
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
    model = RewardModel(model.config, model.transformer, tokenizer)
    print(model)
    logger.info(f"Finished loading model and tokenizer")
    

    # print hf_train_args to see the manually set paras
    print(f"number_train_samples={len(train_dataset)}\nnumber_of_eval_numbers={len(eval_dataset)}")
    print(hf_train_args)
    # raise ValueError("TEST")
    
    ### STEP  5   train
    # trainer = LoRATrainer(
    #     model=model,
    #     args=hf_train_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator = very_clear_data_collator , #modify 不使用这个collator 试试 20230804
    # )
    trainer = RewardTrainer(
    model=model,
    args=hf_train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=rm_data_collator,
       )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(hf_train_args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    train(args)
