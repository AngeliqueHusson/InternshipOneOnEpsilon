"""
    Support vector machine method using the tf-idf feature extraction method
    This file uses as input the file created in the 'Joining and splitting data.py' file.

    @authors Angelique Husson & Nikki Leijnse
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as plt1
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from Feature_extraction import x_train_tfidf1, vectorizer1

# Directory and data import
# Change to your own directory
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Obtaining training and test data
training = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")
trainingBig = pd.read_csv("trainingbig.csv")
category_id_df = pd.read_csv("category_id_df.csv")

# Feature extraction method = tf-idf method
# From import x_train_tfidf1, x_train_tfidf, vectorizer, vectorizer1

# Support vector machine method
clf = LinearSVC(C=1.2).fit(x_train_tfidf1, trainingBig['y_trainBig'])
predicted = clf.predict(vectorizer1.transform(test['x_test']))
print(predicted == test['y_test'])
print(predicted[:20])

# Printing accuracies
result = clf.score(vectorizer1.transform(test['x_test']), test['y_test'], sample_weight=None)
print("The score of Support Vector Machines is: " + str(result))

# One label is not used, different length
id_to_category = dict(category_id_df[['y_id', 'newHashtag']].values) # Dictionary connecting id to hashtag
keys = np.unique(test['y_test'])  # Only get existing id's
target_names = list( map(id_to_category.get, keys))  # Connect existing id's to hashtags
print(metrics.classification_report(test['y_test'], predicted, target_names=target_names))

# Convert to latex table
metrics_result = metrics.classification_report(test['y_test'], predicted, target_names=target_names, output_dict=True)
df = pd.DataFrame(metrics_result).transpose()
df.to_latex('Latex/ResultsSVM_Final.tex', index=True, float_format="%.3f")

# Confusion matrix, does not work correctly yet
conf_mat = confusion_matrix(test['y_test'], predicted)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.newHashtag.values, yticklabels=category_id_df.newHashtag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()