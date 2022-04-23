import pandas as pd
import numpy as np

with open("./qa_dev.txt", 'r+', encoding='utf-8') as f:
    s = [i[:-1].split(',') for i in f.readlines()]

csv = []
for i in range(len(s)):
    csv.append(s[i][0])
csv = pd.DataFrame(csv)
csv.to_csv("./temp_dev.csv")