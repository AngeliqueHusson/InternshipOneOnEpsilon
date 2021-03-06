"""
    Feature extraction methods
"""
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

# Directory and data import
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Obtaining training and validation data
training = pd.read_csv("training.csv")
trainingBig = pd.read_csv("trainingbig.csv")
fullData = pd.read_csv("HashtagText.csv")

# tf - idf method original
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(training['x_train'])
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

vectorizer1 = CountVectorizer()
x_train_counts1 = vectorizer1.fit_transform(trainingBig['x_trainBig'])
tfidf_transformer1 = TfidfTransformer()
x_train_tfidf1 = tfidf_transformer1.fit_transform(x_train_counts1)

filename = 'Webpage/vectorizer1.sav'
pickle.dump(vectorizer1, open(filename, 'wb'))

vectorizer2 = CountVectorizer()
x_train_counts2 = vectorizer2.fit_transform(fullData['text'])
tfidf_transformer2 = TfidfTransformer()
x_train_tfidf2 = tfidf_transformer2.fit_transform(x_train_counts2)

# tf - idf method word level
#vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
#x_train_tfidf = vectorizer.fit_transform(training['x_train'])

#vectorizer1 = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
#x_train_tfidf1 = vectorizer.fit_transform(trainingBig['x_trainBig'])

# tf - idf method n-gram level
#vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#x_train_tfidf = vectorizer.fit_transform(training['x_train'])

#vectorizer1 = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#x_train_tfidf1 = vectorizer.fit_transform(trainingBig['x_trainBig'])

# print('Feature selection matrix:')
# print(x_train_tfidf[:2, :1000])
# print(x_train_tfidf.shape)
# print(vectorizer.get_feature_names())