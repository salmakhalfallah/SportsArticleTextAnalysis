import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import metrics

# importing dataset
df = pd.read_csv('SportsArticleDataset.csv')
print(df.head())

print(df.shape)

# dropping unnecessary/non-numeric values from the dataset
df_filtered = df.drop(['URL'], axis = 1)
y = df_filtered['Label']
X = df_filtered.drop(['TextID', 'Label'], axis = 1)
print(df_filtered.head())

# dropping duplicate labels
X.drop_duplicates()
print(X)
print(y)

# subsetting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

## i would like to look into naive bayes now !
# i am going to look into naive bayes because it is often used for text classification,
# which is what my project entails

# i want to keep in mind the key point that naive bayes assumes that all features are independent of each other

# since we are working with text classification, i will be using multinomial naive bayes
# since the features are discrete counts

from sklearn.naive_bayes import MultinomialNB
naive = MultinomialNB()
naive.fit(X_train, y_train)
y_pred = naive.predict(X_test)
y_pred_train = naive.predict(X_train)
accuracy = naive.score(X_test, y_test)

print(accuracy)