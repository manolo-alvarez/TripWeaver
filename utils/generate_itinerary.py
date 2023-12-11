import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
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
max_new_tokens = 1024
top_p = 1.0
temperature=1.0
city = "Paris"
days = 3

model_name = './experiments/dpo/final_checkpoint/final_merged_checkpoint'
#model_name = '././models/dpo/beta_0.1/final_checkpoint/final_merged_checkpoint'
#model_name = 'meta-llama/Llama-2-7b-hf'
#adapters_name = './experiments/dpo/final_merged_checkpoint'
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

tokenizer = LlamaTokenizer.from_pretrained('./experiments/dpo/final_checkpoint')
tokenizer.bos_token_id = 1
stop_token_ids = [0]

#prompt = "I will be traveling to Austin, United States from 2025-07-15 through 2025-07-17.\n\nGenerate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $1130 total (not including flights and accommodation).\n\nLimit your response to 700 words."
#prompt = "I will be traveling to San Francisco, United States from 2025-03-03 through 2025-03-06. Generate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $1152 total (not including flights and accommodation). Limit your response to 700 words."
#prompt = "I will be traveling to Barcelona, Spain from 2025-07-02 through 2025-07-06. Generate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $1249 total (not including flights and accommodation). Limit your response to 700 words."
#prompt = "I will be traveling to Istanbul, Turkey from 2025-02-18 through 2025-02-20.\n\nGenerate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $540 total (not including flights and accommodation).\n\nLimit your response to 700 words."
prompt = "I will be traveling to Amsterdam, Netherlands from 2024-11-09 through 2024-11-12.\n\nGenerate an itinerary that details activities, transportation between destinations, approximate expenses per activity, breakfast location, lunch location, dinner location, is flexible, and doesn't cost more than $746 total (not including flights and accommodation).\n\nLimit your response to 700 words."
#test = "Day 1: 2024-07-27\n\n8:00 AM - Arrive at Austin-Bergstrom International Airport\nTransportation: Take a ride-share to your accommodation\nExpense: $30\n\n9:00 AM - Check in to your accommodation and freshen up\n\n10:00 AM - Breakfast at Easy Tiger Bake Shop & Beer Garden\nExpense: $15\n\n11:00 AM - Visit the Texas State Capitol\nTransportation: Walking from your accommodation\nExpense: Free\n\n1:00 PM - Lunch at Franklin Barbecue\nExpense: $25\n\n2:30 PM - Explore Lady Bird Lake\nTransportation: Walking from Franklin Barbecue\nExpense: Free\n\n5:00 PM - Check out the Ann and Roy Butler Hike-and-Bike Trail\nTransportation: Walking from Lady Bird Lake\nExpense: Free\n\n7:00 PM - Dinner at Eddie V's Prime Seafood\nExpense: $50\n\nDay 1 Total Expense: $120\n\nDay 2: 2024-07-28\n\n9:00 AM - Breakfast at Counter Cafe East\nExpense: $20\n\n10:00 AM - Visit the Blanton Museum of Art\nTransportation: Ride-share from your accommodation\nExpense: $15\n\n12:00 PM - Lunch at Hillside Farmacy\nExpense: $30\n\n1:30 PM - Explore the Rainey Street Historic District\nTransportation: Walking from Hillside Farmacy\nExpense: Free\n\n4:00 PM - Visit the South Congress Avenue for shopping and live music\nTransportation: Ride-share from Rainey Street\nExpense: Free\n\n7:00 PM - Dinner at Uchi Austin\nExpense: $60\n\nDay 2 Total Expense: $125\n\nDay 3: 2024-07-29\n\n10:00 AM - Breakfast at Kerbey Lane Cafe\nExpense: $15\n\n11:00 AM - Explore the Bullock Texas State History Museum\nTransportation: Walking from Kerbey Lane Cafe\nExpense: $20\n\n1:00 PM - Lunch at Torchy's Tacos\nExpense: $15\n\n2:30 PM - Visit the University of Texas at Austin\nTransportation: Ride-share from Torchy's Tacos\nExpense: Free\n\n5:00 PM - Explore the 2nd Street District for shopping and dining\nTransportation: Walking from the University of Texas at Austin\nExpense: Free\n\n7:00 PM - Dinner at Lamberts Downtown BBQ\nExpense: $40\n\nDay 3 Total Expense: $90\n\nDay 4: 2024-07-30\n\n9:00 AM - Breakfast at Bouldin Creek Cafe\nExpense: $20\n\n10:00 AM - Visit the LBJ Presidential Library\nTransportation: Walking from Bouldin Creek Cafe\nExpense: $20\n\n12:00 PM - Lunch at Home Slice Pizza\nExpense: $25\n\n1:30 PM - Explore Zilker Park\nTransportation: Ride-share from Home Slice Pizza\nExpense: Free\n\n4:00 PM - Visit Barton Springs Pool\nTransportation: Walking from Zilker Park\nExpense: $5 (entrance fee)\n\n7:00 PM - Dinner at Moonshine Patio Bar & Grill\nExpense: $50\n\nDay 4 Total Expense: $120\n\nDay 5: 2024-07-31\n\n9:00 AM - Breakfast at Jo's Coffee\nExpense: $15\n\n10:00 AM - Visit the Austin Aquarium\nTransportation: Ride-share from Jo's Coffee\nExpense: $25\n\n12:00 PM - Lunch at Loro\nExpense: $30\n\n1:30 PM - Explore the Domain for shopping and dining\nTransportation: Ride-share from Loro\nExpense: Free\n\n5:00 PM - Depart for the airport\nTransportation: Ride-share from the Domain\nExpense: $30\n\nDay 5 Total Expense: $100\n\nTotal Expenses for the Trip: $555\n\nThis itinerary is flexible, allowing for adjustments based on personal preferences and any unexpected changes. While the total expenses for activities, meals, and transportation are estimated at $555, it is important to note that additional expenses for souvenirs, tips, and any extra activities are not included in this approximation. As such, it is recommended to keep these potential additional expenses in mind when budgeting for the trip. Overall, this itinerary provides a comprehensive and enjoyable experience in Austin, United States, without exceeding the specified total budget of $912."

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
    with open('experiments/itineraries/itinerary_llama2-7b_DPO_Amsterdam_temp-1.0_top-p-1.0_1.txt', 'w') as f:
        f.write(text)

    return text

generate(model)