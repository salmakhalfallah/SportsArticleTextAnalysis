import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, f1_score

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

# plotting confusion matrix of standard KNN fit
cm = confusion_matrix(y_test, y_pred, labels  = KNN.classes_)
print(cm)

labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
categories = y.unique()
make_confusion_matrix(cm, group_names = labels, categories = categories, cmap = 'Blues', title = 'KNN Confusion Matrix', figsize= (4,4))
plt.show()

# plotting confusion matrix of normalized KNN fit
cm_norm = confusion_matrix(y_test, y_pred_norm, labels  = KNN.classes_)
print(cm_norm)

labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
categories = y.unique()
make_confusion_matrix(cm_norm, group_names = labels, categories = categories, cmap = 'Blues', title = 'KNN Normalized Confusion Matrix', figsize= (4,4))
plt.show()

