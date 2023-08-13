"""
Mainly copied from https://github.com/lvwerra/trl/blob/main/examples/stack_llama/scripts/rl_training.py
Some changes:

用法
CUDA_VISIBLE_DEVICES=0,1,2 python rl_training.py \
    --base_model_name  \
    --merged_sft_model_path ckps/baichaun-sft-hc3-merged \
    --sft_model_lora_path ../weights/hc3_chatgpt_zh_specific_qa_baichuan-7B \
    --reward_model_lora_path ../weights/baichuan-7B_beyond_reward_lora_chinese \
    --adafactor False \
    --save_freq 10 \
    --output_max_length 256 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --batched_gen True \
    --ppo_epochs 4 \
    --seed 0 \
    --learning_rate 1e-5 \
    --early_stopping True \
    --output_dir weights/baichaun_rlhf_beyond_chinese_test_6 \
    --log_with wandb


"""




from accelerate import Accelerator
from tqdm import tqdm
from trl.core import LengthSampler
import argparse
import json
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from transformers import DataCollatorWithPadding, BatchEncoding
from trl import AutoModelForCausalLMWithValueHead , PPOConfig, PPOTrainer, set_seed, PreTrainedModelWrapper
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
import datasets
from datasets import load_dataset,load_from_disk
from peft import (
        PeftConfig,
        PeftModel,
        LoraConfig, 
        TaskType, 
        get_peft_model, 
        prepare_model_for_int8_training, 
        prepare_model_for_kbit_training,
        set_peft_model_state_dict
       )
import bitsandbytes as bnb
from transformers import (
    Adafactor, 
    pipeline, 
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)
from transformers.utils import PaddingStrategy
from transformers import LlamaForSequenceClassification, LlamaConfig, LlamaTokenizer
from transformers import AutoModelForSeq2SeqLM , AutoModel



tqdm.pandas()




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


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    # model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default="", metadata={"help": "the base model name/path"})
    merged_sft_model_path: Optional[str] = field(default="", metadata={"help": "merged_sft_model_path"})
    # tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    sft_model_lora_path: Optional[str] = field(default="", metadata={"help": "the SFT model LoRA path"})
    reward_model_lora_path: Optional[str] = field(default="", metadata={"help": "the Reward model LoRA path"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.5,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=False, metadata={"help": "Use adaptive KL control, otherwise linear"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
# reward_model_name = script_args.reward_model_name


# train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
# train_dataset = load_from_disk('../data/rlhf-reward-single-round-trans_chinese', split='train')
# train_dataset = train_dataset.select(range(100000))


tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name, trust_remote_code=True)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

# tokenizer.pad_token = tokenizer.eos_token
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# training dataset
#dataset = load_from_disk('./data/rlhf-reward-single-round-trans_chinese')
dataset = datasets.load_dataset("beyond/rlhf-reward-single-round-trans_chinese", cache_dir="./dataset")
dataset = dataset['train']
dataset = dataset.select(range(500))
original_columns = dataset.column_names
num_proc = 1

def preprocess_function(examples):
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    # for question in examples["question"]:
    #     query = "Question: " + question + "\n\nAnswer: "
    #     tokenized_question = tokenizer(query, truncation=True)
    #     new_examples["query"].append(query)
    #     new_examples["input_ids"].append(tokenized_question["input_ids"])
    
    # rlhf-reward-single-round-trans_chinese:
    for question in examples["prompt"]:
        query = "问：" + question + "\n\n答："
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])
    return new_examples

dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
dataset = dataset.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
dataset.set_format(type="torch")



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.base_model_name,        #script_args.merged_sft_model_path, # 没啥用，不会加载对应模型
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    project_kwargs={"logging_dir": "ppo_logs"} ,  # https://huggingface.co/docs/trl/logging#logging
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index
print('Loading base model for ppo training...')

"""
下面是原版 StackLLaMA 的实现，是在merge了STF LoRA的模型的基础上，再新增一个LoRA，挺费劲的。
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['W_pack']
)
print('Loading base model for ppo training...')
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=False,
    device_map="auto",
    # device_map={"": current_device},
    peft_config=lora_config,
    trust_remote_code=True
)
"""
# 下面改成不需要merge的方式，直接在SFT LoRA的基础上继续训练:

# load the base model
# base_model_for_PPO = AutoModelForCausalLM.from_pretrained(
#     script_args.base_model_name,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16, 
#     device_map='auto'
#     )
q_config = BitsAndBytesConfig(load_in_4bit= True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)

base_model_for_PPO = AutoModel.from_pretrained(
                                          script_args.base_model_name,
                                          trust_remote_code=True,
                                                                    
                                          load_in_4bit=True,#global_args.load_in_4bit,
                                          torch_dtype=torch.float16,
                                          quantization_config=q_config,
                                          empty_init=False,   # https://github.com/THUDM/ChatGLM-6B/issues/530
                                          #device_map=new_hf_device_map,
                                          # device_map="auto"   # add 20230713
                                     )

base_model_for_PPO.lm_head = base_model_for_PPO.transformer.output_layer
logger.info("prepare_model_for_kbit_training...")
base_model_for_PPO = prepare_model_for_kbit_training(base_model_for_PPO, use_gradient_checkpointing=True)

# install the lora modules
target_modules = find_all_linear_names(base_model_for_PPO)
lora_config = LoraConfig(   
        # AdaLoraConfig 和 qlora好像有冲突 或者是多卡有冲突
        r=64,#global_args.lora_rank,
        lora_alpha=32,#global_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,#global_args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
base_model_for_PPO = get_peft_model(base_model_for_PPO, lora_config)

ckpt = script_args.sft_model_lora_path
adapters_ckpt = os.path.join( ckpt, 'adapter_model.bin' )
adapters_weights = torch.load(adapters_ckpt)  # 这里能看出adapters_Weigth 其实就是个字典
        
logger.info(f"adapter_weights.keys()={adapters_weights.keys()}")
set_peft_model_state_dict(base_model_for_PPO, adapters_weights)
logger.error(f"lora model complete")

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_for_PPO) 
v_head_ckpt = os.path.join(ckpt, 'value_head.bin')
v_head_weights = torch.load(v_head_ckpt)
logger.error(f"v_head_weights={v_head_weights}")
## 注意这里是 model.v_head而非model直接loald 此时model已经是AutoModelForCausalLMWithValueHead , model.v_head和model.pretrained_model是平级的
ppo_model.v_head.load_state_dict(v_head_weights, strict=False) # 

# base_model_for_PPO_with_sft_lora = PeftModel.from_pretrained(
#     base_model_for_PPO, 
#     script_args.sft_model_lora_path
#     )

# wrap with the AutoModelForCausalLMWithValueHead wrapper
# ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     base_model_for_PPO_with_sft_lora
# )

# make the lora modules trainable
for name, param in ppo_model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True


optimizer = None
if script_args.adafactor:  #未使用
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer

# 不显式的传入ref_model 让ppo_trainer自己选择ppo_model的前面若干层当ref_model
# ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     base_model_for_PPO,#script_args.merged_sft_model_path,
#     trust_remote_code=True
# )
ppo_trainer = PPOTrainer(
    config,
    ppo_model, # model with value head
    ref_model=None ,#ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

"""
# 下面这段代码是将reward model直接merge到原模型中，然后通过pipeline来加载。
# 但我希望 reward model依然以 LoRA 的形式存在，因此这里不使用这样的方式
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)
"""
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


# from modeling_baichuan_for_cls import BaichuanForSequenceClassification
# from peft import PeftModel
# print('Loading base model for reward model...')
# base_model_for_RM = BaichuanForSequenceClassification.from_pretrained(
#     script_args.base_model_name, num_labels=1, 
#     trust_remote_code=True, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto",
#     # device_map={"": current_device},
# )
# reward_model = PeftModel.from_pretrained(base_model_for_RM, script_args.reward_model_lora_path)


q_config = BitsAndBytesConfig(load_in_4bit= True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)

reward_model = AutoModel.from_pretrained(        script_args.base_model_name,
                                          trust_remote_code=True,
                                                                    
                                          load_in_4bit=True , #global_args.load_in_4bit,
                                          torch_dtype=torch.float16,
                                          quantization_config=q_config,
                                          empty_init=False,   # https://github.com/THUDM/ChatGLM-6B/issues/530
                                          #device_map=new_hf_device_map,
                                          # device_map="auto"   # add 20230713
                                     )

reward_model.lm_head = reward_model.transformer.output_layer
logger.info("prepare_reward_model_for_kbit_training...")
reward_model  = prepare_model_for_kbit_training(base_model_for_PPO, use_gradient_checkpointing=True)

# install the lora modules
target_modules = find_all_linear_names(reward_model )
lora_config = LoraConfig(   
        # AdaLoraConfig 和 qlora好像有冲突 或者是多卡有冲突
        r=64,#global_args.lora_rank,
        lora_alpha=32,#global_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,#global_args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
reward_model  = get_peft_model(reward_model , lora_config)

ckpt = script_args.reward_model_lora_path
adapters_ckpt = os.path.join( ckpt, 'adapter_model.bin' )
adapters_weights = torch.load(adapters_ckpt)  # 这里能看出adapters_Weigth 其实就是个字典
        
#logger.info(f"adapter_weights.keys()={adapters_weights.keys()}")
set_peft_model_state_dict(reward_model , adapters_weights)
logger.error(f"lora reward_model complete")

reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_model ) 
v_head_ckpt = os.path.join(ckpt, 'value_head.bin')
v_head_weights = torch.load(v_head_ckpt)
logger.error(f"v_head_weights={v_head_weights}")
## 注意这里是 model.v_head而非model直接load 此时model已经是AutoModelForCausalLMWithValueHead , model.v_head和model.pretrained_model是平级的
reward_model.v_head.load_state_dict(v_head_weights, strict=False) # 
logger.error("reward model complete!")


# 然后需要一个得到 reward value 的函数
def get_reward_value(texts):
    #output = reward_model(**tokenizer(texts, return_tensors='pt', padding=True, truncation=True))
    #scores = torch.sigmoid(output.logits).view(-1).tolist()
    #return scores
    _, _, values = reward_model(**tokenizer(texts, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True, return_dict=True)
    rewards = [reward for reward in values[:, -1].float().detach().cpu()] # use fp32 type
    logger.error(f"inside get_reward_value ,rewards={rewards}")
    return rewards

#https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/2f2fd55d8175eb3c6ce94bc821ab4e6331f79d8e/src/llmtuner/tuner/ppo/trainer.py#L172C1-L187C22
@torch.no_grad()  
def get_rewards(
    queries: List[torch.Tensor],
    responses: List[torch.Tensor]
     ) -> List[torch.Tensor]:
    r"""
    Computes scores using given reward model.
    """
    logger.error(f"queries={queries}")
    logger.error(f"responses={responses}")
    batch = ppo_trainer.prepare_model_inputs(queries=queries, responses=responses) 
    _, _, values = reward_model(**batch, output_hidden_states=True, return_dict=True)
    #rewards = [reward for reward in values[:, -1].float().detach().cpu()] # use fp32 type
    rewards = values[-1]  # https://github.com/valkryhx/chatGLM-6B-QLoRA/blob/main/rm_3.py#L820C30-L820C40
    return rewards

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    # "top_k": 0.0,
    # "top_p": 0.95,
    "repetition_penalty": 1.1,
    "do_sample": True,
    #"begin_suppress_tokens": [tokenizer.eos_token_id],  #很重要 但是chatglm2的generate没有这个参数
    
    # "remove_invalid_values": True,
    # "pad_token_id": tokenizer.pad_token_id,
    # "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512
    # "eos_token_id": 100_000, # why？
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {  #并没有用上
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]
    logger.error(f"question_tensors : {question_tensors}")
    
        # """
        # generate这一步经常会报一个奇奇怪怪的bug：
        # RuntimeError: probability tensor contains either inf, nan or element < 0
        # 主要是在model.generate的时候设置了 do_sample=True 就容易报错，但是报错具有随机性，可能在不同的iteration报
        # 关闭 do_sample=True 就不会报错。
        # 可能有用的issue：
        # https://github.com/huggingface/transformers/issues/15169
        # https://github.com/huggingface/transformers/issues/23413
        # https://github.com/huggingface/transformers/issues/22914
        
        # 目前可能的解决办法：
        # 1. 不使用随机采用： do_sample=False，这个基本不会报错，但是感觉影响PPO的性能
        # 2. do_sample=True 的同时，设置 remove_invalid_values=True 参数...还是会报错...奇了怪，而且是报错之后，模型似乎就崩了，一直输出inf,nan了
        
        # update:
        # 发现似乎是由于模型在迭代之后，开始输出空值，而reward却很大，导致模型越学越坏，直接崩了，后面全输出空
        
        # - 百川-base之前推荐的是设置repetition_penalty=1.1，前面没有设置，导致输出很容易重复，而这种输出居然也可以得高分，
        # 因此这里改成一样的配置，目前观察下来有了一些缓解，但后面还是会越学越坏；
        
        # 继续观察，发现当某一次回复为空得到很高的reward之后(得来0.8 的高分，其他的都是0.6的水平)，下一次生成的时候就挂了；
        
        # - 尝试降低learning rate，从 1.4e-5 降低到 1e-5。这个似乎有些效果，可以延缓模型崩溃，但渐渐地回复会越来越短，最终输出空值，属于慢性死亡了。。。
        
        # - 尝试提高 init_kl_coef，从0.2到0.5，也不管用；
        
        # - 继续尝试设置 begin_suppress_tokens 参数，禁止在开头的时候生成 eos token... ！！这是目前最有效的办法了 模型基本不崩了。
        
        # 其实可以发现，主要是reward model太差了，导致对某些不好的输出类型产生了高reward，然后模型就越学越差然后崩了。所以可能问题关键就在于reward model的质量吧。
        
        # """
    response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            # length_sampler=output_length_sampler,  # 这个参数，跟 generation_kwargs 中的 max_new_tokens 只用设置一个
            **generation_kwargs,
        )
    logger.error(f"response_tensors : {response_tensors}")
    # skip_special_tokens=True 就去掉了response中左边的pad_token. 这样下面的q+r连接才不会在中间出现pad_token
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)  
       

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    
    """下面两行是使用pipeline来做，但我这里不采用这种方式
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
    """
    #scores = get_reward_value(texts)
    scores = get_rewards(question_tensors , response_tensors)
    logger.error("we are at line 543")
    rewards = [torch.tensor(score - script_args.reward_baseline) for score in scores]
    for q, r, s in zip(batch["query"], batch["response"], scores):
        print(epoch,'query:',q)
        print('response:',r)
        print('score:',s)
        
    # Run PPO step
    logger.error("we are at line 551")
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
            
    # except Exception as e:
    #     logger.error("we are at line 559 :Exception")
    #     print('---------------------')
    #     print(f"Exception={e}")
    #     print(f"epoch={epoch}")
    #     print(f"question_tensors={question_tensors}")
    #     print('---------------------')
    #     break
