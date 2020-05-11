"""
    Random Forest Method using the tf-idf feature extraction method
    This file uses as input the file created in the 'Joining and splitting data.py' file.

    @authors Angelique Husson & Nikki Leijnse
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from Feature_extraction import x_train_tfidf, vectorizer

# Directory and data import
# Change to your own directory
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Feature extraction method = tf - idf method
# From import x_train_tfidf1, x_train_tfidf, vectorizer, vectorizer1

# Obtaining training and validation data
training = pd.read_csv("training.csv")
validation = pd.read_csv("validation.csv")
category_id_df = pd.read_csv("category_id_df.csv")

# Feature extraction
# From import x_train_tfidf, vectorizer

estimate = []
max_features = round(math.sqrt(x_train_tfidf.shape[1]))  # max_features = sqrt(number of features)
# Random forest
# Tuning Hyperparameters
search_space = [Integer(1, 500, name='n_estimators'), Integer(1,100, name='max_depth')]

@use_named_args(search_space)
def evaluate_model(**params):
    model = RandomForestClassifier() # random_state=0
    model.set_params(**params)
    clf = model.fit(x_train_tfidf, training['y_train'])
    predicted = clf.predict(vectorizer.transform(validation['x_val']))
    estimate = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
    return 1-estimate

result = gp_minimize(evaluate_model, search_space)

# summarizing finding:
print('Best Accuracy: %.3f' % (1-result.fun))
print('Best Parameters: n_estimators=%d, max_depth=%d' % (result.x[0], result.x[1]))
# Best found result n_estimators = 100, max_depth = 39, accuracy = 0.59. max_depth is expected to be good if equal sqrt(#features)

# Prune the tree a little bit it can get better
# Best found result n_estimators = 100, max_depth = 39, accuracy = 0.59. max_depth is expected to be good if equal sqrt(#features)
# Generally the tree gets better if you prune away the bottom and make n_estimators as high as possible
# clf = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=39, random_state=0).fit(x_train_tfidf, training['y_train'])
clf = RandomForestClassifier(n_estimators=result.x[0],max_depth =result.x[1], random_state=77).fit(x_train_tfidf, training['y_train'])
predicted = clf.predict(vectorizer.transform(validation['x_val']))
#print("The out of bag error equals: %.3f" % clf.oob_score_)
print(predicted == validation['y_val'])
print(predicted[:20])

# Printing accuracies
result = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
print("The score of Random Forest is: " + str(result))
# One label is not used, different length
id_to_category = dict(category_id_df[['y_id', 'newHashtag']].values) # Dictionary connecting id to hashtag
keys = np.unique(validation['y_val'])  # Only get existing id's
target_names = list( map(id_to_category.get, keys))  # Connect existing id's to hashtags
print(metrics.classification_report(validation['y_val'], predicted, target_names=target_names))

# Convert to latex table
metrics_result = metrics.classification_report(validation['y_val'], predicted, target_names=target_names, output_dict=True)
df = pd.DataFrame(metrics_result).transpose()
df.to_latex('Latex/ResultsRF.tex', index=True, float_format="%.3f")

# Confusion matrix, does not work correctly yet
conf_mat = confusion_matrix(validation['y_val'], predicted)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.newHashtag.values, yticklabels=category_id_df.newHashtag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()