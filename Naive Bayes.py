"""
    Naive Bayes Method using the tf-idf feature extraction method
    This file uses as input the file created in the 'JoinTextTohashtag.py' file.

    @authors Angelique Husson & Nikki Leijnse
"""

# Naive Bayes method
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Directory and data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

### Naive Bayes
# Feature space = tf - idf method

# Obtaining training and validation data
training = pd.read_csv("training.csv")
validation = pd.read_csv("validation.csv")

# Feature extraction
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(training['x_train'])
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Feature selection matrix:')
print(x_train_tfidf[:2, :1000])
print(x_train_tfidf.shape)
print(vectorizer.get_feature_names())

# Naive Bayes model
clf = MultinomialNB().fit(x_train_tfidf, training['y_train'])
predicted = clf.predict(vectorizer.transform(validation['x_val']))
print(predicted == validation['y_val'])
print(predicted[:20])

result = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
print("The score of Multinomial Naive Bayes is: " + str(result))