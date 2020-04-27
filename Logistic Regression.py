# Logistic Regression method
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from Feature_extraction import x_train_tfidf1, x_train_tfidf, vectorizer

# Directory and data import
# Change to your own directory
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

### Logistic Regression
# Feature space = tf - idf method
# Obtaining training and validation data
training = pd.read_csv("training.csv")
validation = pd.read_csv("validation.csv")
trainingBig = pd.read_csv("trainingbig.csv")
category_id_df = pd.read_csv("category_id_df.csv")

# Feature extraction
# From import x_train_tfidf1, x_train_tfidf, vectorizer, vectorizer1

# Logistic Regression
clf = LogisticRegression(random_state=0).fit(x_train_tfidf, training['y_train'])
predicted = clf.predict(vectorizer.transform(validation['x_val']))
print(predicted == validation['y_val'])
print(predicted[:20])

# Printing accuracies
result = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
print("The score of Multinomial Logistic Regression is: " + str(result))
# One label is not used, different length
id_to_category = dict(category_id_df[['y_id', 'newHashtag']].values) # Dictionary connecting id to hashtag
keys = np.unique(validation['y_val'])  # Only get existing id's
target_names = list( map(id_to_category.get, keys))  # Connect existing id's to hashtags
print(metrics.classification_report(validation['y_val'], predicted, target_names=target_names))

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
    accuracies = cross_val_score(LogisticRegression(random_state=0), x_train_tfidf1, trainingBig['y_trainBig'], scoring='accuracy', cv=i)
    model_name = "Logistic Regression"

    for fold_idx, accuracy in enumerate(accuracies):
        entries.append(("", fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=[model_name, 'fold_idx', 'accuracy'])
    results.append(cv_df.accuracy.mean())

plt.plot(results)
plt.show()
print(results)

sns.boxplot(x= model_name, y='accuracy', data=cv_df)
sns.stripplot(x= model_name, y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print("The accuracy of the Logistic Regression model using Cross Validation is: " + str(cv_df.accuracy.mean()))