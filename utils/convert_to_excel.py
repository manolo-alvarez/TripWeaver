import pandas as pd
from datasets import load_dataset

dataset = load_dataset("json", data_files="././data/itineraries_reduced.json", split="train")
df = pd.DataFrame.from_dict(dataset, orient='columns')
df.to_excel("itineraries_reduced.xlsx", index=False)