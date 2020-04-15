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

df = data[['youtubeVideoId', 'newHashtags', 'text']]
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
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['newHashtags'], random_state=0)

X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

predicted = clf.predict(vectorizer.transform(X_test))
print(predicted == y_test)
print(predicted)

result = clf.score(vectorizer.transform(X_test), y_test, sample_weight=None)
print("The score of Multinomial Naive Bayes is: " + str(result))
