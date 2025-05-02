# testing file

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset
df = pd.read_csv('SportsArticleDataset.csv')
print(df.head())

print(df.shape)

df_filtered = df.drop(['URL'], axis = 1)
X = df_filtered.drop(['TextID', 'Label'], axis = 1)
print(df_filtered.head())

X.drop_duplicates()

# description of numerical data
print(X.describe())

# note some extremely high variances are present, specifically with the totalWordsCounts, CD, DW, MD, PRP, POS, VB, commas, fullstops, past
print(X.var())

# variable skewing
print(X.skew())
