"""
    Finding correlations between labels

    @authors Angelique Husson & Nikki Leijnse
"""

# Naive Bayes method
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Directory and data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Obtaining training and validation data
df = pd.read_csv("trainingbig.csv")

df['y_id'] = df['y_trainBig'].factorize()[0]
category_id_df = df[['y_trainBig', 'y_id']].drop_duplicates().sort_values('y_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['y_id', 'y_trainBig']].values)
print(df.head())

df.to_csv('testing_id.csv')

# Feature extraction
vectorizer = CountVectorizer()
# x_train_counts = vectorizer.fit_transform(df['x_trainBig'])
tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2')
features = tfidf.fit_transform(df.x_trainBig).toarray()
labels = df.y_id
print(features.shape)

N = 2
for y_trainBig, y_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == y_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(y_trainBig))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

