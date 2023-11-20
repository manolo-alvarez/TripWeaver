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
max_new_tokens = 256
top_p = 0.9
temperature=0.7
city = "Paris"
days = 3

model_name = 'meta-llama/Llama-2-13b-hf'
#adapters_name = 'timdettmers/guanaco-13b'
adapters_name, _ = get_last_checkpoint('outputs')

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

prompt = ("""So, you’re planning a trip to Paris? Welcome to one of our favorite cities. With 3 days in Paris, you have enough time to take in the view from the Eiffel Tower, say hi to Mona Lisa in the Louvre, eat street food crepes, climb the Arc de Triomphe, explore several Parisian neighborhoods, and visit the amazing Palace of Versailles. We put together this 3-day Paris itinerary to help you have the best experience.
    ABOUT THIS PARIS ITINERARY
    Paris is one of Europe’s most popular destinations and with that, lines can be long to visit the more popular sites. The last thing you want to do on vacation is to wait in line after line after line. What fun would that be?
    I put a lot of research into how to skip the lines at these attractions. You will have to book some tickets in advance but it will save you hours once you are in Paris. All of the links to book your tickets are included in this post.
    To avoid museum fatigue, I didn’t put too many museums in one day. Yes, the art museums in Paris are amazing, but most people can’t handle more than two museums per day. I know I can’t!
    Finally, all of the times in the daily schedule are rough estimates, just to give you an idea about timing throughout the day. Your times may differ, based on queues and how much time you decide to spend at each place. I did my best to anticipate waiting times and visiting times, but on very busy days (or very quiet days) these times can differ.
    If you only have two days in Paris, check out our 2 Days in Paris Itinerary. 
    Table of Contents

    BEST THINGS TO DO WITH 3 DAYS IN PARIS
    Below is a list of the places to visit if you have 3 days in Paris. All of these are included on this Paris itinerary.
    • Eiffel Tower
    • Arc de Triomphe
    • The Louvre
    • The Champs-Élysées
    • Notre Dame Cathedral
    • Île de la Cité
    • Seine River
    • Musée d’Orsay
    • Montmartre
    • Sacre-Coeur Basilica
    • Versailles
    • The Catacombs
    • Rodin Museum
    • Père Lachaise Cemetery
    """
)

def generate(model, city, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(city=city, days=days), return_tensors="pt").to('cuda')

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