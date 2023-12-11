# 0. imports
import os
import torch

from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer
from trl.import_utils import is_xpu_available

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the location of the base model"})
    merge_adapters: Optional[bool] = field(default=True, metadata={"help": "whether to merge the LoRA adapters in adapter_path with the base model"})
    adapter_path: Optional[str] = field(default="./experiments/sft/final_checkpoint", metadata={"help": "the location of the LoRA adapters"})
    tokenizer_path: Optional[str] = field(default="./experiments/sft/final_checkpoint", metadata={"help": "the location of the tokenizer"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    test_size: Optional[float] = field(default=0.1, metadata={"help": "the ratio of test split to total dataset size"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=100, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    data_path: Optional[str] = field(default="./data/itineraries_reduced.json", metadata={"help": "the data path"})
    output_dir: Optional[str] = field(default="./experiments/dpo", metadata={"help": "the output directory"})
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

def get_itineraries(data_path: str, test_size: int = 0.1) -> Dataset:
    """Load the Itineraries dataset from local and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    args:
        data_path : str : path to dataset
        test_size : int : ratio of test split to total dataset size.

    returs:
        dataset : dict : dataset formatted for training with DPO
    """

    dataset = load_dataset("json", data_files=data_path, split="train")
    # split dataset into train and test
    dataset = dataset.train_test_split(test_size=test_size, shuffle=True)
    # select only the columns that are used for training
    dataset = dataset.select_columns(['prompt','winning_itinerary_content','losing_itinerary_content'])
    # rename columns to be able to use the default DPODataCollatorWithPadding data collator
    dataset = dataset.rename_columns({'winning_itinerary_content':'chosen','losing_itinerary_content':'rejected'})

    return dataset['train'], dataset['test']

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

if __name__ == "__main__":

    # 0. Parse the arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. Load the base model
    print(f"Starting to load the model {script_args.model_path} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map={'':torch.cuda.current_device()}
    )
    model.config.use_cache = False
    print(f"Successfully loaded the model {script_args.model_path} into memory")

    # 2. Merge the adapters with the base_model
    if script_args.merge_adapters:
        print(f"Starting to merge the adapters {script_args.adapter_path} with the model")
        model = PeftModel.from_pretrained(model, script_args.adapter_path, device_map={'':torch.cuda.current_device()})
        model = model.merge_and_unload()
        print(f"Succesfully merged the adapters {script_args.adapter_path} with the model")

    # 3 Ignore bias buffers if using torch distributed
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 4. Load the reference model (same as sft model)
    '''model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map={'':torch.cuda.current_device()}
    )''' # we don't have the reference model from which the dataset was generated (GPT-3.5)

    # 5. Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # 6. Prepare model for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # 6. Create Lora config
    config = LoraConfig(
        r=script_args.lora_r, 
        lora_alpha=script_args.lora_alpha, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=script_args.lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    # 7. Wrap the base_model with the Lora Layer
    model = get_peft_model(model, config)
    model.config.use_cache = False # silence_warnings; TODO check if moving this before SFTTrainer results in the model sucking

    print_trainable_parameters(model)

    # 8. Load the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 9. Load the paired dataset
    train_dataset, eval_dataset = get_itineraries(data_path=script_args.data_path, test_size=script_args.test_size)

    # 10. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2-sft",
        logging_dir=script_args.output_dir
    )

    # 11. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        #model_ref, # we don't have the reference model from which the dataset was generated (GPT-3.5)
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 12. train
    dpo_trainer.train()

    # 13. Save model
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.save_model(output_dir)