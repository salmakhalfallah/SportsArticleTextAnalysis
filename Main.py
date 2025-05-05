import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

from sklearn.tree import DecisionTreeClassifier, plot_tree

# keeping track of scores
scores = []
best_score = 0
best_fit = None

# i want to see how the depth of the tree affects the accuracy of the model
# i am storing the best model and score as well as all the scores for plotting

for d in range(1,20):
    dtree = DecisionTreeClassifier(random_state = 42, max_depth=d)
    dtree.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, dtree.predict(X_test))
    scores.append(score)
    if score > best_score:
        best_score = score
        best_fit = dtree

# printing scores, the best score, and the depth of the best score
print("Scores: ")
print(scores)
print("Best score: ")
print(best_score)
print("Depth of best score: ")
print(scores.index(best_score) + 1)

# plotting the scores against the depth of the tree, where does the chart peak?
plt.plot(range(1,20), scores)
plt.xlabel('max depth')
plt.ylabel('score')
plt.title('Decision Tree Scores Against Depth')
plt.show()

# plotting the decision tree of the best fit
plt.figure(figsize=(20,10))
plot_tree(best_fit, filled = True, feature_names = X.columns, class_names = y.unique())
plt.title('Best DTree Fit')
plt.show()

# plotting feature importance in the decision tree

plt.figure(figsize=(20,10))
plt.bar(X.columns, best_fit.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Importance of Features in DTree')
plt.show()

print("Most important feature: ")
print(X.columns[np.argmax(best_fit.feature_importances_)])
print("Value of most important feature: ")
print(np.argmax(best_fit.feature_importances_))

print("Least important feature: ")
print(X.columns[np.argmin(best_fit.feature_importances_)])
print("Value of least important feature: ")
print(np.argmin(best_fit.feature_importances_))
# would applying random forest improve the model's accuracy?
# i want to look at how important the number of estimators is in the random forest

from sklearn.ensemble import RandomForestClassifier

rfc_scores = []
best_rfc_score = 0
best_rfc_fit = None

for n in range(1, 100):
    rfc = RandomForestClassifier(n_estimators = n, random_state = 42)
    rfc.fit(X_train, y_train)

    score = metrics.accuracy_score(y_test, rfc.predict(X_test))
    rfc_scores.append(score)

    if(score > best_rfc_score):
        best_rfc_score = score
        best_rfc_fit = rfc

print("RFC Scores: ")
print(rfc_scores)
print("Best RFC Score: ")
print(best_rfc_score)
print("Number of Estimators for Best RFC Score: ")
print(rfc_scores.index(best_rfc_score) + 1)

plt.plot(range(1, 100), rfc_scores)
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('RFC Scores Against Number of Estimators')
plt.show()

# plotting the decision tree of the best fit
plt.figure(figsize=(20,10))
plot_tree(best_rfc_fit.estimators_[0], filled = True, feature_names = X.columns, class_names = y.unique())
plt.show()

# plotting feature importances
# important for answering our question of what features are most important in our dataset

plt.figure(figsize=(20,10))
plt.bar(X.columns, best_rfc_fit.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Important of Features in RFC')
plt.show()

print("Most important feature: ")
print(X.columns[np.argmax(best_rfc_fit.feature_importances_)])
print("Value of most important feature: ")
print(np.argmax(best_rfc_fit.feature_importances_))

print("Least important feature: ")
print(X.columns[np.argmin(best_rfc_fit.feature_importances_)])
print("Value of least important feature: ")
print(np.argmin(best_rfc_fit.feature_importances_))