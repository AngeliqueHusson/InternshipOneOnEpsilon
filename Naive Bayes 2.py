# Naive Bayes method
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

# Naive Bayes import
from time import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Directory and data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)
data = pd.read_csv("HashtagText.csv")  # This file has ID, hashtag and text

df = data[['youtubeVideoId', 'newHashtag', 'text']]
print(df[:20])
print(df.shape)

# Feature space
# Document frequency measure, information gain
corpus = []

for i in df['text']:
    corpus.append(i)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.shape)

counts = X.toarray()
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts).toarray()

print(tfidf)
print(tfidf.shape)

### Naive Bayes
# You do not use the previously made tfidf matrix, so it seems unnecessary to make it?
training = pd.read_csv("training.csv")
validation = pd.read_csv("validation.csv")

x_train_counts = vectorizer.fit_transform(training['x_train'])
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

clf = MultinomialNB().fit(x_train_tfidf, training['y_train'])

predicted = clf.predict(vectorizer.transform(validation['x_val']))
print(predicted == validation['y_val'])
print(predicted)

result = clf.score(vectorizer.transform(validation['x_val']), validation['y_val'], sample_weight=None)
print("The score of Multinomial Naive Bayes is: " + str(result))
