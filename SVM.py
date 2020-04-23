# Logistic Regression method
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from Joining_and_splitting_data import category_id_df

# Directory and data import
directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
os.chdir(directory)

### Logistic Regression
# Feature space = tf - idf method

# Obtaining training and validation data
training = pd.read_csv("training.csv")
validation = pd.read_csv("validation.csv")
trainingBig = pd.read_csv("trainingbig.csv")

# Feature extraction
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(training['x_train'])
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

vectorizer1 = CountVectorizer()
x_train_counts1 = vectorizer1.fit_transform(trainingBig['x_trainBig'])
tfidf_transformer1 = TfidfTransformer()
x_train_tfidf1 = tfidf_transformer1.fit_transform(x_train_counts1)

# Logistic Regression
clf = LinearSVC().fit(x_train_tfidf, training['y_train'])
predicted = clf.predict(vectorizer.transform(validation['x_val']))
print(predicted == validation['y_val'])
print(predicted[:20])

# Printing accuracies
result = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
print("The score of Multinomial Logistic Regression is: " + str(result))
# Error, one label is not used, different length
print(metrics.classification_report(validation['y_val'], predicted)) # , target_names=category_id_df.newHashtag.unique()

# Confusion matrix, does not work correctly yet
conf_mat = confusion_matrix(validation['y_val'], predicted)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.newHashtag.values, yticklabels=category_id_df.newHashtag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Cross validation

entries = []

accuracies = cross_val_score(LinearSVC(), x_train_tfidf1, trainingBig['y_trainBig'], scoring='accuracy')

model_name = "SVM"

for fold_idx, accuracy in enumerate(accuracies):
  entries.append(("", fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=[model_name, 'fold_idx', 'accuracy'])

sns.boxplot(x= model_name, y='accuracy', data=cv_df)
sns.stripplot(x= model_name, y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print(cv_df.accuracy.mean())