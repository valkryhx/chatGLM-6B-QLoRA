#!python
# -*- coding: utf-8 -*-
# @author: hx



from transformers import DataCollatorWithPadding, BatchEncoding
from trl import AutoModelForCausalLMWithValueHead
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from glob import glob
from loguru import logger
import os
import torch
import evaluate
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union,Sequence,Tuple

from datasets import load_dataset
from peft import (
        PeftModel,
        LoraConfig, 
        TaskType, 
        get_peft_model, 
        prepare_model_for_int8_training, 
        prepare_model_for_kbit_training,
       )
import bitsandbytes as bnb
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from transformers.utils import PaddingStrategy
from transformers import LlamaForSequenceClassification, LlamaConfig, LlamaTokenizer
from transformers import AutoModelForSeq2SeqLM , AutoModel

# from reward_model import RewardModel

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



IGNORE_INDEX = -100
class DataCollatorForChatGLM(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        ignore_pad_token_for_loss: Optional[bool] = False
    ):
        super().__init__(tokenizer, padding=True)
        self.model = model
        self.label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id
        if tokenizer.eos_token_id == 130005:
            self.get_attention_masks = self.get_attention_masks_v1
            self.get_position_ids = self.get_position_ids_v1
        else:
            self.get_attention_masks = self.get_attention_masks_v2
            self.get_position_ids = self.get_position_ids_v2

    def get_attention_masks_v1(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.

        Note that ChatGLM assigns False on token to be attended in attention mask. In general settings, it should be True.

        According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L680
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()

        for i, seq in enumerate(input_ids):
            attention_mask[i, :, :(seq == self.tokenizer.bos_token_id).nonzero()[0].item()] = 1 # context
            attention_mask[i, :, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0 # padding

        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    def get_position_ids_v1(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates position ids for left-padded sequenes.

        According to: https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L692
        """
        batch_size, seq_length = input_ids.size()
        mask: int = self.model.config.mask_token_id
        gmask: int = self.model.config.gmask_token_id
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        block_position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        for i, seq in enumerate(input_ids):
            mask_token = gmask if gmask in seq else mask
            context_length = (seq == self.tokenizer.bos_token_id).nonzero()[0].item()
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(
                seq_length - padding_length,
                dtype=torch.long,
                device=device
            )
            if self.model.position_encoding_2d or (mask_token != gmask): # 2d position encoding or not gMASK
                position_ids[i, context_length:] = (seq == mask_token).nonzero()[0].item() - padding_length # mask position
            block_position_ids[i, context_length:] = torch.arange(
                seq_length - context_length,
                dtype=torch.long,
                device=device
            ) + 1

        if self.model.position_encoding_2d:
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)

        return position_ids

    def get_attention_masks_v2(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length), device=device)

        for i, seq in enumerate(input_ids):
            attention_mask[i, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0 # padding

        return attention_mask

    def get_position_ids_v2(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates position ids for left-padded sequenes.
        """
        batch_size, seq_length = input_ids.size()
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        for i, seq in enumerate(input_ids):
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(seq_length - padding_length, dtype=torch.long, device=device)

        return position_ids

    def __call__(self, features: Sequence[Dict[str, Any]]) -> BatchEncoding:
        r"""
        Pads batched data to the longest sequence in the batch.

        We adopt left-padding in both training and evaluation.
        """
        if isinstance(features[0]["input_ids"], torch.Tensor):
            input_ids = [feature["input_ids"].clone().detach().flip(0) for feature in features]
        else:
            input_ids = [torch.tensor(feature["input_ids"]).flip(0) for feature in features]

        if "labels" in features[0]:
            if isinstance(features[0]["labels"], torch.Tensor):
                labels = [feature["labels"].clone().detach().flip(0) for feature in features]
            else:
                labels = [torch.tensor(feature["labels"]).flip(0) for feature in features]
            input_ids += labels # pad them to the same length

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).flip(-1)

        batch = {}

        if "labels" in features[0]:
            input_ids, labels = input_ids.split(len(features), dim=0)
            labels = torch.where(labels != self.tokenizer.pad_token_id, labels, self.label_pad_token_id)
            batch["labels"] = labels

        batch["input_ids"] = input_ids
        batch["attention_mask"] = self.get_attention_masks(input_ids, device=input_ids.device)
        batch["position_ids"] = self.get_position_ids(input_ids, device=input_ids.device)

        return BatchEncoding(batch)



class PairwiseDataCollatorForChatGLM(DataCollatorForChatGLM):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """

        features = [{"input_ids": feature[key]} for key in ("accept_ids", "reject_ids") for feature in features]
        return super().__call__(features)

                          

class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        print("in PairWiseLoss Func")
        print(f"【in PairWiseLoss chosen_reward】={chosen_reward}")
        print(f"【in PairWiseLoss reject_reward】={reject_reward}")
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs + 1e-7)  # add to avoid inf loss
        loss = -log_probs.mean()
        print(f"【in PairWiseLoss  log_probs】={log_probs}")
        print(f"【in PairWiseLoss  loss】={loss}")
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
            print(f"seq_len, batch_size, hidden_size = hidden_states.shape={hidden_states.shape}")
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
        #rewards = rewards.mean(dim=-1)
        #if len(rewards.shape) == 2:
        #    rewards = rewards.squeeze(1)    # ensure shape is (B)

        #assert len(rewards.shape) == 1 and rewards.shape[0] == batch_size

        ## 为了适应方法2/方法3 直接返回 rewards = self.v_head(hidden_states).squeeze(-1) 
        return rewards

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
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
        #print(f"input_ids={input_ids}")
        #print(f"attention_mask={attention_mask}")
        #print(f"chosen_input_ids={chosen_input_ids}")
        #print(f"rejected_input_ids={rejected_input_ids}")
        if input_ids is not None and attention_mask is not None :
            
            #方法1 使用全部的token的hiddenstate来计算loss
            # total_reward = self.reward(input_ids ,attention_mask=attention_mask , position_ids=None)
            # half = input_ids.shape[0]//2
            # chosen_reward = total_reward[:half]
            # reject_reward = total_reward[half:]
            # loss = self.loss_fn(chosen_reward, reject_reward)

            # 方法2 使用reward tensor   最后一个来token（不会是pad 因为chatglm2的pad side为left 最后一个token就是eos）计算loss 的reward
            # 参考了 https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/src/glmtuner/tuner/rm/trainer.py#L43
            total_reward = self.reward(input_ids ,attention_mask=attention_mask , position_ids=None)
            print(f"input_ids.shape={input_ids.shape}")
            print(f"total_reward.shape={total_reward.shape}")
            batch_size = input_ids.size(0) // 2
            print(f"batch_size={batch_size}")
            # 注意是 total_reward[:,-1] 而不是  total_reward[-1] 前面保留了维度dim=0和dim=1最后一列 后者是dim=1全保留但是dim=0最后一行
            chosen_reward, reject_reward = total_reward[:,-1].split(batch_size, dim=0)
            loss = -torch.log(torch.sigmoid(chosen_reward - reject_reward) + 1e-7).mean()
            #logger.error(f"use new method2,loss ={loss}")

            #方法3 找到最后一个非pad token的EOStoken来计算loss  参考了deepspeed-chat 注意tokenizer的padding size为right
            # https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py#L55
            # 实际上 pad side为left  那最后一个就是eos 也就不需要像方法3这样费劲去找最后一个非pad字符了
            # 并且chatglm2不支持设置pad side为right
            # tokenization_chatglm.py:228  assert self.padding_side == "left" 
            # self.num_padding_at_beginning = 0 # for all models except OPT(=1) so we keep this parameter
            # rewards = self.reward(input_ids ,attention_mask=attention_mask , position_ids=None)
            # chosen_mean_scores = []
            # rejected_mean_scores = []

            # # Split the inputs and rewards into two parts, chosen and rejected
            # assert len(input_ids.shape) == 2
            # bs = input_ids.shape[0] // 2
            # seq_len = input_ids.shape[1]

            # chosen_ids = input_ids[:bs]  # bs x seq x 1
            # rejected_ids = input_ids[bs:]
            # chosen_rewards = rewards[:bs]
            # rejected_rewards = rewards[bs:]

            # # Compute pairwise loss. Only backprop on the different tokens before padding
            # loss = 0
            # for i in range(bs):
            #     chosen_id = chosen_ids[i]
            #     rejected_id = rejected_ids[i]
            #     chosen_reward = chosen_rewards[i]
            #     rejected_reward = rejected_rewards[i]

            #     c_inds = (chosen_id == self.pad_id).nonzero()
            #     c_ind = c_inds[self.num_padding_at_beginning].item() if len(
            #     c_inds
            #     ) > self.num_padding_at_beginning else seq_len  # self.num_padding_at_beginning =0 here,OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            #     check_divergence = (chosen_id != rejected_id).nonzero()

            #     if len(check_divergence) == 0:
            #         end_ind = rejected_reward.size(-1)
            #         divergence_ind = end_ind - 1
            #         r_ind = c_ind
            #     else:
            #         # Check if there is any padding otherwise take length of sequence
            #         r_inds = (rejected_id == self.pad_id).nonzero()
            #         r_ind = r_inds[self.num_padding_at_beginning].item(
            #         ) if len(r_inds) > self.num_padding_at_beginning else seq_len  # self.num_padding_at_beginning =0
            #         end_ind = max(c_ind, r_ind)
            #         divergence_ind = check_divergence[0]
            #     assert divergence_ind > 0
            #     c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            #     r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            #     chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
            #     rejected_mean_scores.append(rejected_reward[r_ind - 1])

            #     loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

            # loss = loss / bs
            # chosen_reward = torch.stack(chosen_mean_scores)
            # reject_reward = torch.stack(rejected_mean_scores)
        #     return {
        #     "loss": loss,
        #     "chosen_reward": chosen_reward,
        #     "reject_reward": reject_reward,
        # }
      
            
            # 注意下面的chosen_reward/reject_reward 都是sigmoid之后的 在[0,1]之间 不是真正的reward函数返回的reward 那个reward在[-无穷，+无穷]
            # return {
            # "loss": loss,
            # "chosen_reward": torch.sigmoid(chosen_reward) + 1e-7 if chosen_reward is not None else chosen_reward,
            # "reject_reward": torch.sigmoid(reject_reward) + 1e-7 if reject_reward is not None else reject_reward,
            #   }
            # reward 不使用sigmoid变换 直接存 原始计算值
            return {
                "loss": loss,
                "chosen_reward": chosen_reward if chosen_reward is not None else chosen_reward,
                "reject_reward": reject_reward if reject_reward is not None else reject_reward ,
            }
            
            
            
        if chosen_input_ids is not None and rejected_input_ids is not None :
            logger.error("both not None")
            total_ids = torch.cat((chosen_input_ids,rejected_input_ids),dim=0)
            total_attention_mask = torch.cat((chosen_attention_mask,rejected_attention_mask),dim=0)
            total_position_ids = torch.cat((chosen_position_ids, rejected_position_ids),dim=0)
            total_reward = self.reward(total_ids ,attention_mask=total_attention_mask , position_ids=total_position_ids)
            assert chosen_input_ids.shape[0]  == rejected_input_ids.shape[0] 
            half = chosen_input_ids.shape[0] 
            chosen_reward = total_reward[:half]
            reject_reward = total_reward[half:]
            loss = self.loss_fn(chosen_reward, reject_reward)
            logger.error(f"use new method,loss ={loss}")
            return {
            "loss": loss,
            "chosen_reward": torch.sigmoid(chosen_reward) +1e-7 if chosen_reward is not None else chosen_reward,
            "reject_reward": torch.sigmoid(reject_reward) +1e-7 if reject_reward is not None else reject_reward,
              }
        
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
            "chosen_reward": torch.sigmoid(chosen_reward)+1e-7 if chosen_reward is not None else chosen_reward,
            "reject_reward": torch.sigmoid(reject_reward)+1e-7 if reject_reward is not None else reject_reward,
        }

def preprocess_function(examples,tokenizer,max_length=512):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
        "accept_ids":[],
        "reject_ids":[]
    }
    # for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
    for question, response_j, response_k in zip(examples["user_input"], examples["completion_a"], examples["completion_b"]):
        chatglm2_prompt = "[Round 1]\n\n问：{}\n\n答：{}\n\n"
        tokenized_j = tokenizer(
            #"Question: " + question + "\n\nAnswer: " + response_j, truncation=True
             chatglm2_prompt.format(question,response_j),truncation=True,padding='max_length',max_length=max_length
              )
        tokenized_k = tokenizer(
            #"Question: " + question + "\n\nAnswer: " + response_k, truncation=True
              chatglm2_prompt.format(question,response_k),truncation=True,padding='max_length',max_length=max_length
             )

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

        new_examples["accept_ids"].append(tokenized_j["input_ids"])
        new_examples["reject_ids"].append(tokenized_k["input_ids"])
    return new_examples



def get_rm_datset(data_path, tokenizer, max_samples=-1,max_length=512,global_args=None):
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
                                lambda example: preprocess_function(example, tokenizer=tokenizer,max_length=max_length),
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

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

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
            #padding=self.padding,
            padding='max_length',
            #truncation=True,
            
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            #padding=self.padding,
            padding='max_length',
            #truncation=True,
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
            "input_ids": torch.cat((batch_j["input_ids"], batch_k["input_ids"]),dim=0)  ,
            "attention_mask": torch.cat((batch_j["attention_mask"] , batch_k["attention_mask"]),dim=0),
        }
        return batch



@dataclass
class RewardDataCollatorWithPadding_only_input_ids:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

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
            #padding=self.padding,
            padding='max_length',
            #truncation=True,
            
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            #padding=self.padding,
            padding='max_length',
            #truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            #"input_ids_j": batch_j["input_ids"],
            #"attention_mask_j": batch_j["attention_mask"],
            #"input_ids_k": batch_k["input_ids"],
            #"attention_mask_k": batch_k["attention_mask"],
            #"return_loss": True,
            "input_ids": torch.cat((batch_j["input_ids"], batch_k["input_ids"]),dim=0)  ,
            #"attention_mask": torch.cat((batch_j["attention_mask"] , batch_k["attention_mask"]),dim=0),
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def compute_accuracy(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, _ = eval_preds
    return {"accuracy": (preds[0] > preds[1]).sum() / len(preds[0])}


# https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/src/glmtuner/extras/save_and_load.py#L15
def get_state_dict(model: torch.nn.Module, trainable_only: Optional[bool] = True) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if (not trainable_only) or v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict

VALUE_HEAD_FILE_NAME = "value_head.bin"

TRAIN_TYPE = "lora"
class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        # print('inputs["input_ids_j"]: ', inputs["input_ids_j"].shape)
        # print('inputs["attention_mask_j"]: ', inputs["attention_mask_j"].shape)
        #rewards_j = model.forward(
        #    chosen_input_ids=inputs["input_ids_j"], chosen_attention_mask=inputs["attention_mask_j"])["chosen_reward"]
        # print("rewards_j: ", type(rewards_j), rewards_j.shape)

        # print('inputs["input_ids_k"]: ', inputs["input_ids_k"].shape)
        # print('inputs["attention_mask_k"]: ', inputs["attention_mask_k"].shape)
        #rewards_k = model.forward(
        #    rejected_input_ids=inputs["input_ids_k"], rejected_attention_mask=inputs["attention_mask_k"])["reject_reward"]
        # print("rewards_k: ", type(rewards_k), rewards_k.shape)

        ### 上面的写法会导致rewardmodel的forward一会有chosen 一会没chosen
        res = model.forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        #loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return res["loss"], {"rewards_j": res["chosen_reward"], "rewards_k": res["reject_reward"]}
        print({"rewards_j": res["chosen_reward"], "rewards_k": res["reject_reward"]})
        return res["loss"]
    
    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)
        print(model)
        if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
            logger.error(" 111  model, pretrained_model")
            backbone_model = getattr(model, "pretrained_model")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
        else:
            backbone_model = model

        if isinstance(backbone_model, PeftModel): # LoRA tuning
            logger.error("222 backbone_model, PeftModel")
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
        elif isinstance(backbone_model, PreTrainedModel): 
            logger.error("333 backbone_model, PreTrainedModel")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
            #print(f'hasattr(backbone_model, "lora_A")={hasattr(backbone_model, "lora_A")}') 这个要具体的layer才行 否则用model级别判定是False的
            if TRAIN_TYPE =="lora" :
                print(f"TRAIN_TYPE ==lora")
                backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
            elif TRAIN_TYPE !="lora" : #  freeze/full-tuning or p_tuning
                print(f"TRAIN_TYPE !=lora")
                #backbone_model.config.use_cache = True
                backbone_model.save_pretrained(
                    output_dir,
                    state_dict=get_state_dict(backbone_model, trainable_only=False),
                    safe_serialization=self.args.save_safetensors
                )
            backbone_model.config.use_cache = False
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            logger.warning("No model to save.")
            
    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     """只保存adapter"""
    #     print("begin to save  !!!")
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     if self.is_world_process_zero():  
    #         self.model.save_pretrained(output_dir)
    #         torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
    #         print("save done !!!")
    #     else :
    #         print("this process is not main process , do not save model.[for distributed training scenario]")



class RewardTrainer_trl(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        We use score on the EOS token to represent reward of the whole sentence.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.

        Note that the first element will be removed from the output tuple.

        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """
        batch_size = inputs["input_ids"].size(0) // 2
        ## 根据 源码 https://github.com/lvwerra/trl/blob/main/trl/models/modeling_value_head.py#L184
        ## AutoModelForCausalLMWithValueHead.forard 返回 3个值 return (lm_logits, loss, value) 仅仅 需要最后一个作为reward
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        # 这里是使用reward最后的一个分布来计算loss  
        r_accept, r_reject = values[-1].split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(r_accept - r_reject)+1e-7).mean()
        return (loss, [loss, r_accept, r_reject]) if return_outputs else loss

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
    
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.0001)
    model_name: Optional[str] = field(
        default="decapoda-research/llama-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub or local."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "Enables gradient checkpointing."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The max length of token list "},
    )
    output_dir: Optional[str] = field(
        default="reward_model_output",
        metadata={"help": "The dir of reward_model "},
    )



def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    data_path="./data/rm_data"
    data_files_list = glob(f'{data_path}/**/*.json', recursive=True) + glob(
                f'{data_path}/**/*.jsonl', recursive=True)
          

    
    # train_dataset = load_dataset("json",data_files=data_files_list, split="train")
    # if script_args.train_subset > 0:
    #     train_dataset = train_dataset.select(range(script_args.train_subset))
    
    # eval_dataset = load_dataset("json",data_files=data_files_list, split="train")
    # if script_args.eval_subset > 0:
    #     eval_dataset = eval_dataset.select(range(script_args.eval_subset))

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]

    # output_name = (
    #     f"reward_model_{model_name_split}__{script_args.train_subset}_{script_args.learning_rate}"
    # )
    output_name = script_args.output_dir

    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=10,  # 500,
    save_strategy="steps",
    save_steps=10,  # 500,
    save_total_limit=2,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    # local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    # bf16=script_args.bf16,
    # fp16=True, #! this is important! if True, cuda out of memory.
    logging_strategy="steps",
    logging_steps=2,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to =["tensorboard"]
)

    # Load the value-head model and tokenizer.
    # tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_auth_token=True)
    if "llama" in script_args.model_name or "vicuna" in script_args.model_name or "Vicuna" in script_args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        config = LlamaConfig.from_pretrained(script_args.model_name)

    
    #https://github.com/microsoft/DeepSpeedExamples/issues/338  
    elif "chatglm" in script_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name, trust_remote_code=True ) 
        config = AutoConfig.from_pretrained(
            script_args.model_name, trust_remote_code=True)
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(
            script_args.model_name, trust_remote_code=True)

    print("tokenizer: ", type(tokenizer)) 

    if "llama" in script_args.model_name or "vicuna" in script_args.model_name or "Vicuna" in script_args.model_name:
        # required for llama
        tokenizer.add_special_tokens(
            {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        # required for gpt2
        #tokenizer.pad_token = tokenizer.eos_token
        print(f"tokenizer.pad_token={tokenizer.pad_token}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    print("device_map: ", device_map)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
    # )

    if "llama" in script_args.model_name or "vicuna" in script_args.model_name or "Vicuna" in script_args.model_name:
        model = LlamaForSequenceClassification.from_pretrained(
            script_args.model_name,
            num_labels=1,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
    elif "chatglm" in script_args.model_name:
        q_config = BitsAndBytesConfig(load_in_4bit= True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
    
        model = AutoModel.from_pretrained(
            script_args.model_name,
            #num_labels=1,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map=device_map,
            quantization_config=q_config,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name,
            num_labels=1,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map=device_map,
        
        )

    #add lm_head  只有使用trl  AutoModelForCausalLMWithValueHead 才需要
    #model.lm_head = model.transformer.output_layer
    #print("model: ", type(model))
    #model = RewardModel(model.config, model.transformer, tokenizer)
    #print(model)
    model = prepare_model_for_kbit_training(model)
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    #print("model: ", type(model))


    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,#TaskType.SEQ_CLS,  #https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/src/glmtuner/tuner/core/adapter.py#L87C27-L87C45
        inference_mode=False,
        target_modules = target_modules ,
        r=64,  # for qlora 64 is ok
        lora_alpha=16,  # 32,
        lora_dropout=0.05,  # 0.1,
        bias="none",
    )

    # 参考了 很多例子 都是先peft lora 再转换成rewardmodel
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    model = RewardModel(model.config, model.transformer, tokenizer)
    #model = RewardModel(model.config, model, tokenizer)  
    # 这里直接传model 会在外面包裹好几层 导致 .transformer(XX)调用报错
    #(transformer): PeftModelForCausalLM(
    # (base_model): LoraModel(
    #  (model): ChatGLMForConditionalGeneration(
    # (transformer): ChatGLMModel(
    #model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    print(model)
    print(f"Finished loading model and tokenizer")

   

    train_dataset = get_rm_datset(data_path=data_path, tokenizer=tokenizer, max_samples=script_args.train_subset,max_length=script_args.max_length,global_args=None)
    eval_dataset  = get_rm_datset(data_path=data_path, tokenizer=tokenizer, max_samples=script_args.eval_subset,max_length=script_args.max_length,global_args=None)

    # Train the model.
    trainer = RewardTrainer(
        model=model ,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_accuracy,
        data_collator = RewardDataCollatorWithPadding( tokenizer=tokenizer, max_length=script_args.max_length, pad_to_multiple_of=8), 
        #data_collator=RewardDataCollatorWithPadding_only_input_ids(tokenizer=tokenizer, max_length=script_args.max_length, pad_to_multiple_of=8),
        )

    model.config.use_cache = False

    trainer.train(script_args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    # model.save_pretrained(script_args.output_dir + "peft_last_checkpoint")
    model.save_pretrained(output_name)

if __name__ == "__main__":
    #args = parse_args()
    ## test load ckpt
    q_config = BitsAndBytesConfig(load_in_4bit= True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
    
    model = AutoModel.from_pretrained(
            "THUDM/chatglm2-6b",#script_args.model_name,
            #num_labels=1,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto",#device_map,
            quantization_config=q_config,
        )
    model = prepare_model_for_kbit_training(model)
    
    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(   # AdaLoraConfig 和 qlora好像有冲突 或者是多卡有冲突
        task_type=TaskType.CAUSAL_LM,#TaskType.SEQ_CLS,  #https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/src/glmtuner/tuner/core/adapter.py#L87C27-L87C45
        inference_mode=False,
        target_modules = target_modules ,
        r=64,  # for qlora 64 is ok
        lora_alpha=16,  # 32,
        lora_dropout=0.05,  # 0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    ckpt = "/kaggle/working/chatGLM-6B-QLoRA/reward_model_0809_v1/checkpoint-20"
    checkpoint_name = os.path.join(
                ckpt, 'adapter_model.bin'
            )
    adapters_weights = torch.load(checkpoint_name)
    print(f"adapter_weights={adapters_weights}")
    set_peft_model_state_dict(model, adapters_weights)
    model = RewardModel(model.config, model.transformer, tokenizer)
    v_head_ckpt = os.path.join(
                ckpt, 'value_head.bin'
            )
    v_head_weights = torch.load(v_head_ckpt)
    print(f"v_head_weights={v_head_weights}")
    model.load_state_dict(v_head_weights, strict=False)
    raise ValueError(123)
    train()
