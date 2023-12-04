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

#model_name = 'runs/sft/final_checkpoint/final_merged_checkpoint'
model_name = 'runs/dpo/beta_0.1/final_checkpoint/final_merged_checkpoint'
#model_name = 'meta-llama/Llama-2-7b-hf'
#adapters_name = 'timdettmers/guanaco-13b'
#adapters_name, _ = get_last_checkpoint('./outputs/final_checkpoint')
#adapters_name = 'runs/dpo/beta_0.1/final_checkpoint'

print(f"Starting to load the model {model_name} into memory")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={'':torch.cuda.current_device()}
)
#model = PeftModel.from_pretrained(model, adapters_name, device_map={'':torch.cuda.current_device()})
#model = model.merge_and_unload()
tokenizer = LlamaTokenizer.from_pretrained('runs/dpo/beta_0.1')
tokenizer.bos_token_id = 1

stop_token_ids = [0]
model.eval()

prompt = "I will be traveling to Bangkok,\u00a0Thailand from 2025-08-14 through 2025-08-18.\n\nGenerate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $824 total (not including flights and accommodation).\n\nLimit your response to 700 words."
test = "Day 1: 2025-08-14\n\n8:00 AM: Arrive in Bangkok\nTransportation: Take a taxi from the airport to your accommodation - approximately $20\nAccommodation: check into hotel in central Bangkok\n\n9:00 AM: Breakfast at local caf\u00e9\nLocation: CAF\u00c9 DELICES\nApproximate expense: $10\n\n10:00 AM: Visit the Grand Palace and Wat Phra Kaew\nLocation: Na Phra Lan Rd, Phra Nakhon, Bangkok\nApproximate expense: $20 (including entrance fee and guided tour)\n\n1:00 PM: Lunch at local street food vendor\nLocation: Street food market near Grand Palace\nApproximate expense: $8\n\n3:00 PM: Explore Wat Arun (Temple of Dawn)\nLocation: Chao Phraya River, Bangkok Yai, Bangkok\nApproximate expense: $5 (entrance fee)\n\n6:00 PM: Dinner at Thip Samai Pad Thai\nLocation: 313-315 Maha Chai Rd, Samran Rat, Phra Nakhon, Bangkok\nApproximate expense: $15\n\nDay 1 total approximate expense: $78\n\nDay 2: 2025-08-15\n\n8:00 AM: Breakfast at hotel\nLocation: Hotel dining area\nIncluded in accommodation cost\n\n9:00 AM: Visit Chatuchak Weekend Market\nTransportation: Take the Skytrain to Mo Chit station - approximately $2\nLocation: Kamphaeng Phet 2 Rd, Chatuchak, Bangkok\nApproximate expense: $15 for shopping and snacks\n\n1:00 PM: Lunch at Chatuchak Weekend Market\nLocation: Food stalls within the market\nApproximate expense: $10\n\n3:00 PM: Explore Jim Thompson House\nLocation: Rama I Rd, Wang Mai, Pathum Wan District, Bangkok\nApproximate expense: $10 (entrance fee and guided tour)\n\n6:00 PM: Dinner at Thipsamai Phad Thai Pratu Pie\nLocation: Phra Nakorn Road, Thailand\nApproximate expense: $12\n\nDay 2 total approximate expense: $49\n\nDay 3: 2025-08-16\n\n8:00 AM: Breakfast at hotel\nLocation: Hotel dining area\nIncluded in accommodation cost\n\n9:00 AM: Visit Wat Pho\nLocation: 2 Sanamchai Rd, Grand Palace Subdistrict, Pranakorn District, Bangkok\nApproximate expense: $15 (entrance fee and guided tour)\n\n1:00 PM: Lunch at Tha Tien Boat Noodle\nLocation: 392/9 Maha Rat Rd, Khwaeng Phra Borom Maha Ratchawang, Khet Phra Nakhon, Bangkok\nApproximate expense: $10\n\n3:00 PM: Explore Bangkok National Museum\nLocation: Na Phra That Alley, Phra Borom Maha Ratchawang, Phra Nakhon, Bangkok\nApproximate expense: $10 (entrance fee and guided tour)\n\n6:00 PM: Dinner at Supanniga Eating Room\nLocation: 160/11 Soi Sukhumvit 55, Klongton-Nua, Wattana, Bangkok\nApproximate expense: $20\n\nDay 3 total approximate expense: $55\n\nDay 4: 2025-08-17\n\n8:00 AM: Breakfast at hotel\nLocation: Hotel dining area\nIncluded in accommodation cost\n\n9:00 AM: Day trip to Ayutthaya\nTransportation: Join a guided tour to Ayutthaya - approximately $40\nLocation: Ayutthaya Historical Park\nApproximate expense: $30 (including transportation and entrance fees)\n\n1:00 PM: Lunch at local restaurant in Ayutthaya\nLocation: Restaurant within Ayutthaya Historical Park\nApproximate expense: $12\n\n6:00 PM: Dinner at local Bangkok restaurant\nLocation: Restaurant nearby accommodation\nApproximate expense: $25\n\nDay 4 total approximate expense: $107\n\nDay 5: 2025-08-18\n\n8:00 AM: Breakfast at hotel\nLocation: Hotel dining area\nIncluded in accommodation cost\n\nCheck out of hotel and depart for the airport\nTransportation: Taxi to the airport - approximately $20\n\nDay 5 total approximate expense: $20\n\nTotal approximate expense for the trip: $309\n\nThis itinerary is flexible and allows for additional activities or adjustments based on individual preferences. With a total approximate expense of $309, it falls well within the specified budget of $824."

def generate(model, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt, return_tensors="pt")

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
    with open('itinerary_dpo.txt', 'w') as f:
        f.write(text)

    return text

generate(model)