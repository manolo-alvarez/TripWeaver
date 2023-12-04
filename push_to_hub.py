import pandas as pd
import datasets
from datasets import load_dataset

dataset = load_dataset("json", data_files="test/itinerary_3.json", split="train")
df = pd.DataFrame.from_dict(dataset, orient='columns')
df.to_excel("test/itinerary_3.xlsx", index=False)

exit(-1)
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/itinerary.json", split="train")

# split dataset into train and test
dataset = dataset.train_test_split(test_size=0.1, shuffle=True)

dataset['train'].save_to_disk("test")