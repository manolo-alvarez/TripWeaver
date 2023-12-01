import torch
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer, PretrainedConfig
from trl import DPOTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

# Prefix
prefix = "summarize: "

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

def preprocess_cnn_dailymail_dataset    (examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def return_prompt_and_responses(samples) -> dict[str, str, str]:
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
            for question in samples["question"]
        ],
        "chosen": samples["response_j"],   # rated better than k
        "rejected": samples["response_k"], # rated worse than j
    }

# Base model: meta-llama/Llama-2-13b-hf
model_name = "meta-llama/Llama-2-13b-hf"

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

# Load model with quantization config
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={'':torch.cuda.current_device()})

print(f"Successfully loaded the model {model_name} into memory")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

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
model = get_peft_model(model, config)

print_trainable_parameters(model)

# Load dataset
# run: rm -r ~/.cache/huggingface/datasets if having trouble with disk space.
###################### load DATASET: cnn_dailymail ######################
data = load_dataset("cnn_dailymail", '3.0.0')
data = data.map(preprocess_cnn_dailymail_dataset, batched=True)

################# load DATASET: stack-exchange-paired ###################
'''
data = load_dataset("lvwerra/stack-exchange-paired")
original_columns = data.column_names

data.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)
'''
#########################################################################
# needed for gpt-neo-x tokenizer
# TODO check why this is needed
tokenizer.pad_token = tokenizer.eos_token

# Set data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Load evaluation method
rouge = evaluate.load("rouge")

############### Train model with DPO ###############
dpo_trainer = DPOTrainer(
    model,
    model,
    beta=0.1,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=50,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="none",
        save_steps=10,
        resume_from_checkpoint=True
    )
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

dpo_trainer.train()
dpo_trainer.save_model()

############### Train model with QLORA ###############
'''
# Instantitate trainer with training parameters
trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="none",
        save_steps=10,
        resume_from_checkpoint=True
    )
)

trainer.train()

# Save model config for inference
PretrainedConfig().save_pretrained("outputs")
'''