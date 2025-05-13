import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, f1_score

from KNN import test_accuracy


# credits for confusion matrix code: Dennis Trimarchi

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42)
# next is decision trees... i'm excited for this one !

from sklearn.tree import DecisionTreeClassifier, plot_tree

# keeping track of scores
scores = []
best_score = 0
best_fit = None

# i want to see how the depth of the tree affects the accuracy of the model
# i am storing the best model and score as well as all the scores for plotting

for d in range(1,100):
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
plt.plot(range(1,100), scores)
plt.xlabel('max depth')
plt.ylabel('score')
plt.title('Decision Tree Scores Against Depth')
plt.figure(figsize=(8,6))
plt.show()

# plotting the decision tree of the best fit
plt.figure(figsize=(20,10))
plot_tree(best_fit, filled = True, feature_names = X.columns, class_names = y.unique())
plt.title('Best DTree Fit')
plt.show()

# training and testing scores

train_accuracy = best_fit.score(X_train, y_train)
test_accuracy = best_fit.score(X_test, y_test)

print("Train Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

# plotting the confusion matrix

y_pred = best_fit.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels = best_fit.classes_)
labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
categories = y.unique()
make_confusion_matrix(cm, group_names = labels, categories = categories, cmap = 'Blues', title = 'Decision Tree Confusion Matrix')
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

# testing and training accuracy

train_accuracy_rfc = best_rfc_fit.score(X_train, y_train)
test_accuracy_rfc = best_rfc_fit.score(X_test, y_test)
print("RFC Train Accuracy: ", train_accuracy_rfc)
print("RFC Test Accuracy: ", test_accuracy_rfc)

# plotting confusion matrix

y_pred_RFC = best_rfc_fit.predict(X_test)
cm_RFC = confusion_matrix(y_test, y_pred_RFC, labels = best_rfc_fit.classes_)
make_confusion_matrix(cm_RFC, group_names = labels, categories = categories, cmap = 'Blues', title = 'Random Forest Confusion Matrix')
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
