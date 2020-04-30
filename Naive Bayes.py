"""
    Naive Bayes Method using the tf-idf feature extraction method
    This file uses as input the file created in the 'Joining and splitting data.py' file.

    @authors Angelique Husson & Nikki Leijnse
"""

import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from Feature_extraction import x_train_tfidf1, x_train_tfidf, vectorizer

# Directory and data import
# Change to your own directory
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Feature space = tf - idf method
# From import x_train_tfidf1, x_train_tfidf, vectorizer, vectorizer1

# Obtaining training and validation data
training = pd.read_csv("training.csv")
validation = pd.read_csv("validation.csv")
trainingBig = pd.read_csv("trainingbig.csv")
category_id_df = pd.read_csv("category_id_df.csv")

# Feature extraction
# From import x_train_tfidf1, x_train_tfidf, vectorizer, vectorizer1

# Naive Bayes model
clf = MultinomialNB().fit(x_train_tfidf, training['y_train'])
predicted = clf.predict(vectorizer.transform(validation['x_val']))
print(predicted == validation['y_val'])
print(predicted[:20])

# Printing accuracies
result = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
print("The score of Multinomial Naive Bayes is: " + str(result))
# One label is not used, different length
id_to_category = dict(category_id_df[['y_id', 'newHashtag']].values) # Dictionary connecting id to hashtag
keys = np.unique(validation['y_val'])  # Only get existing id's
target_names = list( map(id_to_category.get, keys))  # Connect existing id's to hashtags
print(metrics.classification_report(validation['y_val'], predicted, target_names=target_names))

# Convert to latex table
metrics_result = metrics.classification_report(validation['y_val'], predicted, target_names=target_names, output_dict=True)
df = pd.DataFrame(metrics_result).transpose()
df.to_latex('Latex/ResultsNB.tex', index=True, float_format="%.3f")

# Confusion matrix, does not work correctly yet
conf_mat = confusion_matrix(validation['y_val'], predicted)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.newHashtag.values, yticklabels=category_id_df.newHashtag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Cross validation
results = []

for i in range(2,20):
    entries = []
    accuracies = cross_val_score(MultinomialNB(), x_train_tfidf1, trainingBig['y_trainBig'], scoring='accuracy', cv=i)
    model_name = "Naive Bayes"

    for fold_idx, accuracy in enumerate(accuracies):
        entries.append(("", fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=[model_name, 'fold_idx', 'accuracy'])
    results.append(cv_df.accuracy.mean())

plt.plot(results)
plt.show()
print(results)

sns.boxplot(x=model_name, y='accuracy', data=cv_df)
sns.stripplot(x=model_name, y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print("The maximum accuracy of the Naive Bayes model using Cross Validation is: " + str(max(results)))
