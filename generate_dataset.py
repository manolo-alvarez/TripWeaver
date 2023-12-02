from openai import OpenAI
import json
import random
import re

def replace_itinerary_placeholders (prompt:str, city:str, start_date:str, end_date:str, budget:int):
  '''
  Replaces the placeholders for city, start_date, end_date, and budget in prompt.

  args:
    prompt : str : prompt used to generate itinerary
    city : str : city of trip
    start_date : str : start date of trip
    end_date : str : end date of trip
    budget : int : total budget for trip

  returns:
    new_prompt : str : the initial prompt with placeholders replaced

  '''
  # Replace the placeholders in the prompt with the random cities, dates, and budget
  new_prompt = re.sub('__city__', city.strip(), prompt)
  new_prompt = re.sub('__start_date__', start_date, new_prompt)
  new_prompt = re.sub('__end_date__', end_date.strip(), new_prompt)
  new_prompt = re.sub('__budget__', str(budget), new_prompt)
  return new_prompt 

def replace_score_placeholders (prompt:str, itinerary:str):
  '''
  Replaces the placeholders for city, start_date, end_date, and budget in prompt.

  args:
    prompt: str : prompt used to generate itinerary
    itinerary : str : itinerary

  returns:
    new_prompt : str : the initial prompt with placeholders replaced

  '''
  # Replace the placeholders in the prompt with the random cities, dates, and budget
  new_prompt = re.sub('__itinerary__', itinerary, prompt)
  return new_prompt                                

## Get prompt files ##
init_itinerary_prompt_file = open("data/itinerary_prompt.txt", "r")
init_itinerary_prompt = init_itinerary_prompt_file.read()
init_itinerary_prompt_file.close()
init_score_prompt_file = open("data/score_prompt.txt", "r")
init_score_prompt = init_score_prompt_file.read()
init_score_prompt_file.close()
cities_file = open("data/cities.txt", "r")
cities = cities_file.readlines()
cities_file.close()
dates_file = open("data/dates.txt", "r")
dates = dates_file.readlines()
dates_file.close()
######################}

# Open data file
data_file = open('data/itinerary.json', 'w')

# Open Client for OpenAI API
client = OpenAI()

for i in range(2):#len(dates):

  # Generate itinerary prompt parameters
  city = cities[i%len(cities)].strip()
  start_date = dates[i].split(",")[0]
  end_date = dates[i].split(",")[1].strip()
  budget = random.randint(500,1500)
  itinerary_prompt = replace_itinerary_placeholders(init_itinerary_prompt, city, start_date, end_date, budget)

  # Generate itineraries 1 and 2
  itinerary_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    n=2, # number of chat completion choices to generate for each input message
    temperature=1.0,
    messages=[
      {"role": "user",
      "content": itinerary_prompt}
    ]
  )

  # Score itinerary 1
  score_1_prompt = replace_score_placeholders(init_score_prompt, itinerary_response.choices[0].message.content)
  score_1_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    n=1, # number of chat completion choices to generate for each input message
    temperature=0, # want a deterministic score
    messages=[
      {"role": "user",
      "content": score_1_prompt}
    ]
  )

  # Score itinerary 2
  score_2_prompt = replace_score_placeholders(init_score_prompt, itinerary_response.choices[1].message.content)
  score_2_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    n=1, # number of chat completion choices to generate for each input message
    temperature=0, # want a deterministic score
    messages=[
      {"role": "user",
      "content": score_2_prompt}
    ]
  )

  # Compare scores of itineraries
  indexOfScore_1 = score_1_response.choices[0].message.content.find("@")+1
  indexOfScore_2 = score_2_response.choices[0].message.content.find("@")+1

  if (indexOfScore_1 > 0) and (indexOfScore_2 > 0):
    try:
      score_1 = int(score_1_response.choices[0].message.content[indexOfScore_1:indexOfScore_1+2])
      score_2 = int(score_2_response.choices[0].message.content[indexOfScore_2:indexOfScore_2+2])
    except:
      print("character after @ is not an integer for data entry #" + str(i))
    else:
      if score_1 > score_2: 
        indexOfWinner = 0
        indexOfLoser = 1
        winning_score = score_1
        losing_score = score_2
      else: 
        indexOfWinner = 1
        indexOfLoser = 0
        winning_score = score_2
        losing_score = score_1
      
      datapoint = {
        "prompt" : itinerary_prompt,
        "winning_itinerary_content": itinerary_response.choices[indexOfWinner].message.content,
        "losing_itinerary_content": itinerary_response.choices[indexOfLoser].message.content,
        "winning_itinerary_score": winning_score,
        "losing_itinerary_score": losing_score,
        "city": city,
        "start_date": start_date,
        "end_date": end_date,
        "total_budget": budget,
        "itinerary_model": itinerary_response.model,
        "score_model": score_1_response.model
      }
      
      # convert the dictionary to a JSON string
      datapoint_json = json.dumps(datapoint)
      # write the JSON string to the file with a new line
      data_file.write(datapoint_json + '\n')

  else:
    print("could not find the score for data entry #" + str(i))

data_file.close()

################## Archive #####################

'''
def save_datapoints(datapoints:list, file_path:str):
  
  creates a file where each line is a new JSON representation of a datapoint from the list of datapoints

  args:
    datapoints : list : list of datapoints
    file_path : str : path to output file
  
  with open(file_path, 'w') as file:
    for datapoint in datapoints:
      # convert the dictionary to a JSON string
      datapoint_json = json.dumps(datapoint)
      # write the JSON string to the file with a new line
      file.write(datapoint_json + '\n')'''