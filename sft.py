import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from datasets import load_dataset, Dataset
from trl.import_utils import is_xpu_available
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import os

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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

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

output_dir = os.path.join("outputs", "final_checkpoint")
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map={'':torch.cuda.current_device()}, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

# Base model: meta-llama/Llama-2-13b-hf
model_name = "meta-llama/Llama-2-7b-hf"

print(f"Starting to load the model {model_name} into memory")

# Set load config to 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer for base model 
# A tokenizer is in charge of preparing the inputs for a model. 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load model with quantization config
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={'':torch.cuda.current_device()})

print(f"Successfully loaded the model {model_name} into memory")

# Enable gradient checkpointing
base_model.gradient_checkpointing_enable()

# Prepare model for k-bit training
base_model = prepare_model_for_kbit_training(base_model)

# Load Lora config
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# Get PEFT model
# Wraps the model with the LoraLayer
base_model = get_peft_model(base_model, config)

print_trainable_parameters(base_model)

# Load dataset
# run: rm -r ~/.cache/huggingface/datasets if having trouble with disk space.
train_dataset, test_dataset = get_itineraries(tokenizer,"data/itinerary.json", test_size=0.1)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=config,
    packing=True,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=TrainingArguments(
            output_dir="outputs",
            max_steps=100,
            logging_steps=10,
            save_steps=10,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False,
            group_by_length=False,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.05,
            optim="paged_adamw_32bit",
            bf16=True,
            remove_unused_columns=False,
            run_name="sft_llama2",
            report_to="none"
    )
)
base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Train model
trainer.train()
trainer.save_model("outputs")

output_dir = os.path.join("outputs", "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map={'':torch.cuda.current_device()}, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)