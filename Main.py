import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(X, y)

# first model is using K-NN
# i'm not expecting great performance due to the amount of features in the dataset (despite large # of training examples)
# it's also easily fooled by irrelevant attributes
# i'm first going to work with KNN w/o normalization to see how the model performs

# training model here
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train, y_train)

y_pred = KNN.predict(X_test)
y_pred_train = KNN.predict(X_train)

from sklearn import metrics

train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
test_accuracy = metrics.accuracy_score(y_test, y_pred)

# 84% training accuracy, 79% test accuracy... not bad. can we do better? let's try k = 20 instead of 10
print(train_accuracy, test_accuracy)

# when k = 20, train acc = 82%, test acc = 81%
# this is better but not that much better

# when k = 30, train acc = 81%, test acc = 79%
# just as i suspected, this isn't the best training model for our data set.

# but let's try normalizing the attributes!

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

KNN.fit(X_train, y_train)
y_pred_norm = KNN.predict(X_test)
y_pred_training_norm = KNN.predict(X_train)

train_accuracy_norm = metrics.accuracy_score(y_train, y_pred_training_norm)
test_accuracy_norm = metrics.accuracy_score(y_test, y_pred_norm)

# not better results with normalization.
# figured as much, since the dimensionality is high for this dataset.

print(test_accuracy_norm, train_accuracy_norm)

# i was going to plot the decision boundary... i literally have too many features </3

# next is decision trees... i'm excited for this one !

