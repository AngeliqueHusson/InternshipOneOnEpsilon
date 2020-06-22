# Retrieved from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from Feature_extraction import x_train_tfidf2

# Define function to plot training and cross validation accuracies for different training sizes
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 20)):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Sample size")
    axes[0].set_ylabel("Accuracy")

    # Determine accuracy for different training sizes
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)

    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Save mean accuracy to dataframe
    table = [train_sizes, train_scores_mean, test_scores_mean]

    # Save all results
    samples = list()
    for s in train_sizes:
        for t in range(0,10):
            samples.append(s)

    acc_train = list()
    for t in train_scores:
        acc_train.extend(t)

    acc_test = list()
    for v in test_scores:
        acc_test.extend(v)

    df = pd.DataFrame({'samples': samples, 'train': acc_train, 'test': acc_test})
    df.to_csv('accuracies.csv')

    return plt, train_sizes, train_scores, test_scores

fig, axes = plt.subplots(1, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

# Directory
directory = 'C:/Users/Nikki/Desktop/Internship AM/Epsilon Project/Data'
os.chdir(directory)
# Retrieve data
alldata = pd.read_csv("HashtagText.csv")
X = x_train_tfidf2
y = alldata['y_id']

# Choose plot title
title = "Learning Curves (Support Vector Machine)"
# Set cross validation parameters
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# Choose the estimator
estimator = SVC(C=1.2)
# Call function to plot training accuracy against cross validation accuracy for different sample sizes
plot_learning_curve(estimator, title, X, y, ylim=(0.1, 1.01),
                    cv=cv, n_jobs=4)
plt.show()



