from openai import OpenAI

'''
client = OpenAI()

system_content = open("system_content.txt", "r")

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  n=2, # number of chat completion choices to generate for each input message
  temperature=1.0,
  messages=[
    {"role": "system", "content": system_content.read()},
    {"role": "user", "content": "Paris, France, 2023-12-30, 2024-01-02, 1500"}
  ]
)

system_content.close()
'''
import json
 
# Opening JSON file
f = open('test.json')
 
# returns JSON object as a dictionary
response = json.load(f)

itinerary_1 = response['ChatCompletion']['choices'][0]['message']['content']
itinerary_2 = response['ChatCompletion']['choices'][1]['message']['content']

with open('itinerary_1.txt', 'w') as f:
    f.write(itinerary_1)

with open('itinerary_2.txt', 'w') as f:
    f.write(itinerary_2)