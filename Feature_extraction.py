"""
    Feature extraction methods
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer


# Directory and data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Obtaining training and validation data
training = pd.read_csv("training.csv")
trainingBig = pd.read_csv("trainingbig.csv")

# tf - idf method
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(training['x_train'])
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

vectorizer1 = CountVectorizer()
x_train_counts1 = vectorizer1.fit_transform(trainingBig['x_trainBig'])
tfidf_transformer1 = TfidfTransformer()
x_train_tfidf1 = tfidf_transformer1.fit_transform(x_train_counts1)

print('Feature selection matrix:')
print(x_train_tfidf[:2, :1000])
print(x_train_tfidf.shape)
print(vectorizer.get_feature_names())