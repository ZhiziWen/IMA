import pandas as pd
import os
import re
import numpy as np

df = pd.read_csv("feature_vector_file.csv")
df = df.iloc[: , 1:]
path = 'resources/modified_classes'
buggy = []

for filename in os.listdir(path):
   with open(os.path.join(path, filename), 'r') as f:
       text = f.read()
       sentences = re.findall('\S+', text)
       for all in sentences:
           clean_text = re.findall('\w+', all)
           buggy.append(clean_text[len(clean_text) - 1])

df["buggy"] = np.where(df['class'].isin(buggy), 1, 0)
df.to_csv("new_feature_vector_file.csv")
print(df)