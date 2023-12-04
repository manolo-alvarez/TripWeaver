import json

# Open the input file and the output file
with open('data/itinerary.json', 'r') as infile, open('test/itinerary_3.json', 'w') as outfile:
    # Read the file line by line
    for line in infile:
        # Parse the JSON object
        obj = json.loads(line)

        # Calculate the difference between the two scores
        difference = abs(obj['winning_itinerary_score'] - obj['losing_itinerary_score'])

        # If the difference is strictly greater than 5, write the object to the output file
        if difference > 5:
            outfile.write(json.dumps(obj) + '\n')