import pandas as pd

samples = pd.read_csv("../dataset/sample_messages.csv")
print(samples.columns.tolist())
print(samples.head(3).to_string())