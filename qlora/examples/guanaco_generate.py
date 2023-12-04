import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaTokenizer
from peft import PeftModel
from peft.tuners.lora import LoraLayer

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


# TODO: Update variables
max_new_tokens = 700
top_p = 0.9
temperature=1.0
city = "Paris"
days = 3

model_name = 'meta-llama/Llama-2-13b-hf'
#adapters_name = 'timdettmers/guanaco-13b'
#adapters_name, _ = get_last_checkpoint('./outputs/final_checkpoint')
adapters_name = './outputs/final_merged_checkpoint'

print(f"Starting to load the model {model_name} into memory")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={'':torch.cuda.current_device()}
)
model = PeftModel.from_pretrained(model, adapters_name)
model = model.merge_and_unload()
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1

stop_token_ids = [0]
model.eval()

prompt = "I will be traveling to Paris, France from 2024-07-20 through 2024-07-23. Generate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $1000 total (not including flights and accommodation). Limit your response to 700 words."

def generate(model, city, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model, city)
import pdb; pdb.set_trace()