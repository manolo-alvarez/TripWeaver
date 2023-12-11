import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, HfArgumentParser
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from trl import SFTTrainer
from trl.import_utils import is_xpu_available
from trl.trainer import ConstantLengthDataset
from typing import Optional

@dataclass
class ScriptArguments:
    """
    The arguments for the sft training script.
    """

    # training parameters
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the path to the remote or local base model"})
    tokenizer_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the path to the remote or local base tokenizer"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "optimizer learning rate"})
    test_size: Optional[float] = field(default=0.1, metadata={"help": "the ratio of test split to total dataset size"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=50, metadata={"help": "the number of warmup steps"}) #TODO: test with 10 and 50; original was 100
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})

    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_steps: Optional[int] = field(default=100, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "the evaluation frequency"})

    data_path: Optional[str] = field(default="././data/itineraries.json", metadata={"help": "the data path"})
    output_dir: Optional[str] = field(default="././models/sft", metadata={"help": "the output directory"})
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

def prepare_sample_text(sample):
    """Prepare the text from a sample of the dataset."""
    text = f"Prompt: {sample['prompt']}\n\nItinerary: {sample['chosen']}"
    return text

def chars_token_ratio(dataset, tokenizer, nb_examples=100):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def get_itineraries(tokenizer, data_path: str, test_size: int) -> Dataset:
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

    chars_per_token = chars_token_ratio(dataset['train'], tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        dataset["train"],
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    test_dataset = ConstantLengthDataset(
        tokenizer,
        dataset["test"],
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    return train_dataset, test_dataset

if __name__ == "__main__":
    
    # 0. Parse the arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. Load 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 2. Load model with quantization config
    print(f"Starting to load the model {script_args.model_path} into memory")
    base_model = AutoModelForCausalLM.from_pretrained(script_args.model_path, quantization_config=bnb_config, device_map={'':torch.cuda.current_device()})
    print(f"Successfully loaded the model {script_args.model_path} into memory")

    # 3. Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()

    # 4. Prepare model for 4-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # 5. Create Lora config
    config = LoraConfig(
        r=script_args.lora_r, 
        lora_alpha=script_args.lora_alpha, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=script_args.lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    # 6. Wrap the base_model with the Lora Layer
    base_model = get_peft_model(base_model, config)
    base_model.config.use_cache = False # silence_warnings; TODO check if moving this before SFTTrainer results in the model sucking

    print_trainable_parameters(base_model)

    # 7. Load tokenizer for base model 
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # 8. Load dataset
    train_dataset, test_dataset = get_itineraries(tokenizer,script_args.data_path, test_size=script_args.test_size)

    # 9. Initialize training arguments:
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        max_steps=script_args.max_steps,
        eval_steps=script_args.eval_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        group_by_length=False,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        weight_decay=script_args.weight_decay,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="sft_llama2",
        report_to=script_args.report_to,
        logging_dir=script_args.output_dir
    )

    # 10. Initialize the SFT Trainer
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=config,
        packing=True,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args
    )

    # 11. Train model
    trainer.train()

    # 12. Save model
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.save_model(output_dir)