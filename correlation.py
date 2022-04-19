import pandas as pd
from scipy.stats import pearsonr
import numpy as np


fileObj = open("correlationInput.txt", "r")

arr = fileObj.read().splitlines()

a = arr[0].split(" ")

for i in range(len(a)):
    a[i] = float(a[i])
print(a)

b = arr[1].split(" ")

for i in range(len(b)):
    b[i] = float(b[i])


df = pd.DataFrame(dict(x=a))

CORR_VALS = np.array(b)
def get_correlation(vals):
    return pearsonr(vals, CORR_VALS)[0]

df['correlation'] = df.rolling(window=len(CORR_VALS)).apply(get_correlation)

print(df['correlation'].values)