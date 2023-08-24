# The code is inspired by stack_llama_2/scripts/dpo_llama2.py
# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py
# I modified the codes to train chatglm2-6b by DPO algorithm with Qlora.
# Author huangxun
# 20230823

# peft ==0.4.0
# accelerate == 0.21.0
# trl == 0.5.1.dev0 这个版本还是开发态 修正后ref_model可以为None 让trainer自己去根据model创建ref_model 主要是节省memory避免oom
# !pip install git+https://github.com/huggingface/trl

# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy # 用于把model 深拷贝一份 放到另外的gpu上作为ref_model
import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments,BitsAndBytesConfig
import bitsandbytes as bnb
from trl import DPOTrainer
from loguru import logger

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    # 这下面的dataset_name_or_path 暂时还未用上
    dataset_name_or_path: Optional[str] = field(
        default="./data/paired_anli_0823/paired_anli.json",
        metadata={"help": "the location of the dataset json file name or path"},
    )
    
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps and overrides num_train_epochs: if set positive"})
    num_train_epochs: Optional[float] = field(default=3, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=20, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=20, metadata={"help": "the evaluation frequency"})
    save_total_limit:Optional[int] = field(default=None, metadata={"help": "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir"})
    load_best_model_at_end:Optional[bool] = field(default=False, metadata={"help": "Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved."})
    
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_stack_exchange_paired(
    dataset_name_or_path="./data",
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=4, #24
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    # dataset = load_dataset( # 加载公开数据集时使用 可以把公开数据集下载到data_dir中 此时需要带上data_dir参数
    #     "lvwerra/stack-exchange-paired",
    #     split="train",
    #     cache_dir=cache_dir,
    #     data_dir=data_dir,
    # )

    dataset = load_dataset( # 加载本地自己构造的数据集时使用 使用data_files=list[str] 或者 str代表本地的数据集文件名
        "json",
        data_files=dataset_name_or_path,#"data/paired_anli_0823/paired_anli.json",
        split="train",
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.shuffle().select(range(min(len(dataset), 200)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

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

def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class MyDPOTrainer(DPOTrainer): 
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

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model, if it was trained by qlora , remember to add quantization_config in from_pretrained.
    q_config = q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        #device_map='auto',
        quantization_config = q_config, # add q_config here for qlora
        trust_remote_code = True,
        
    ).to("cuda:0")
    # now model is a peftmodel
    model.config.use_cache = False
    model.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    model.enable_input_require_grads()
   
    logger.info("prepare_model_for_kbit_training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) 
    model.print_trainable_parameters()
    
    logger.error("check model layers layout on devices")
    for i in model.named_parameters():
        print(f"{i[0]} -> {i[1].device}")
    
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    # model_ref = AutoPeftModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     #load_in_4bit=True,
    #     device_map='auto',
    #     quantization_config = q_config, # add q_config here for qlora
    #     trust_remote_code = True,
    # )

    # model_ref = AutoPeftModelForCausalLM.from_pretrained(  # 这 已经是一个qlora的peftmodel
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,  
    #     #device_map='auto',
    #     quantization_config = q_config, # add q_config here for qlora
    #     trust_remote_code = True,
        
    # ).to("cuda:1")
    model_ref = copy.deepcopy(model).to("cuda:1")
    torch_gc()
    logger.info(f"id(model)={id(model)}")
    logger.info(f"id(model_ref)={id(model_ref)}")
    # now model is a peftmodel
    model_ref.config.use_cache = False
    #model_ref.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    #model_ref.enable_input_require_grads()

    #logger.info("prepare_model_ref_for_kbit_training...")
    #model_ref = prepare_model_for_kbit_training(model_ref, use_gradient_checkpointing=True) 
   
    
    logger.error("check model layers layout on devices")
    for i in model_ref.named_parameters():
        print(f"{i[0]} -> {i[1].device}")



    
    tokenizer_name_or_path = "THUDM/chatglm2-6b"
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token  
    #chatglm2-6b 不支持这样赋值 况且chatglm2-6b本身的pad_token=<unk> pad_token_id=0  
    # https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L42 中使用的是pad_token_id 这个在chatglm2中已经有值了 =0

    torch_gc()
    
    # 2. Load the Stack-exchange paired dataset
    #train_dataset = get_stack_exchange_paired(data_dir="data/rl", sanity_check=script_args.sanity_check)
    train_dataset = get_stack_exchange_paired(dataset_name_or_path=script_args.dataset_name_or_path, sanity_check=script_args.sanity_check)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )
    logger.info(f"train_dataset={train_dataset}")
    # 3. Load evaluation dataset
    #eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation", sanity_check=True)
    eval_dataset = get_stack_exchange_paired(dataset_name_or_path=script_args.dataset_name_or_path, sanity_check=True)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )
    logger.info(f"eval_dataset={eval_dataset}")
    
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        load_best_model_at_end = script_args.load_best_model_at_end,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        #bf16=True,
        fp16=True,
        num_train_epochs = script_args.num_train_epochs ,
        remove_unused_columns=False,
        run_name="dpo_chatglm2",
    )
    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # target_modules=[  # These modules are for llama2
        #     "q_proj",
        #     "v_proj",
        #     "k_proj",
        #     "out_proj",
        #     "fc_in",
        #     "fc_out",
        #     "wte",
        # ],
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    # ref_model (`PreTrainedModelWrapper`):
    # Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
    # reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L42
    logger.info("prepare my dpo_trainer")
    my_dpo_trainer = MyDPOTrainer(
        model,
        ref_model =model_ref,#None, #model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,  #虽然mode已经是peft model了 但是仍然要用peft_config 指明可训练的modules
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    my_dpo_trainer.train()
    my_dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
