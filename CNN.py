"""
    Naive Bayes Method using the tf-idf feature extraction method
    This file uses as input the file created in the 'JoinTextTohashtag.py' file.

    @authors Angelique Husson & Nikki Leijnse
"""

# Neural Network method
import os
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout, AveragePooling1D, MaxPooling1D, GRU, SimpleRNN, Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from Feature_extraction import x_train_tfidf1, x_train_tfidf, vectorizer, vectorizer1
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

# Directory and data import
# Change to your own directory
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

# Obtaining training and validation data
training = pd.read_csv("training.csv")
print(len(training["x_train"]))
validation = pd.read_csv("validation.csv")
trainingBig = pd.read_csv("trainingbig.csv")
category_id_df = pd.read_csv("category_id_df.csv")

EMBEDDING_DIM = 100
MAX_NB_WORDS = x_train_tfidf1.shape[1]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(trainingBig["x_trainBig"])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(trainingBig["x_trainBig"])
X = pad_sequences(X, maxlen=len(word_index))

# Word vectorization

# model1 = Word2Vec(X, size=300, window=5, min_count=1, workers=4)
#
# vectorizer1 = CountVectorizer()
# x_train_counts1 = vectorizer1.fit_transform(model1)
# tfidf_transformer1 = TfidfTransformer()
# x_train_tfidf1 = tfidf_transformer1.fit_transform(x_train_counts1)
#
# X = x_train_tfidf1
Y = pd.get_dummies(trainingBig["y_trainBig"])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=12)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#Neural Network model
# Embedding layer
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=MAX_NB_WORDS,
              input_length = X.shape[1],
              output_dim=EMBEDDING_DIM))
model.add(AveragePooling1D())
model.add(SpatialDropout1D(0.2))

model.add(Conv1D(128, 5, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.25))#3

# model.add(Dense(10, activation='relu'))

model.add(Dense(11, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

epochs = 50
batch_size = 70

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

acc = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(acc[0], acc[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();