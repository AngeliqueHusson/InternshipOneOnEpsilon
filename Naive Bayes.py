# Naive Bayes method
import re
import os
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after clean 2'

os.chdir(directory)
filelist = os.listdir(directory)

# Feature space
# Document frequency measure, information gain
corpus = []

for i in filelist:
    with open(i, encoding='gb18030', errors='ignore') as file:

        textString = file.read()
        print(textString)
        #words = word_tokenize(textString)
        corpus.append(textString)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.shape)

counts = X.toarray()

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts).toarray()

print(tfidf)
print(tfidf.shape)



# Naive Bayes

# Instead of forming the Feature space from all the text files, we want to make the feature space from the file we made.