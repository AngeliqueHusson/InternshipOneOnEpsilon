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
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout, AveragePooling1D, MaxPooling1D, GRU, SimpleRNN, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from Feature_extraction import x_train_tfidf1, x_train_tfidf, vectorizer
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

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

tokenizer = Tokenizer(num_words=x_train_tfidf1.shape[1], filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(trainingBig["x_trainBig"])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Recurrent Neural Network model
def RNN(EMBEDDING_DIM, NEURONS_LSTM, dropout, dropout1):
    model = Sequential()
    # model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    # Embedding layer
    model.add(Embedding(input_dim=x_train_tfidf1.shape[1],
              input_length = X.shape[1],
              output_dim=EMBEDDING_DIM,
              trainable=True))  # mask_zero=True
    model.add(BatchNormalization())
    model.add(AveragePooling1D())
    # model.add(SpatialDropout1D(0.2))

    # Long short-term memory layer
    model.add(LSTM(NEURONS_LSTM, return_sequences=True,
               dropout=dropout, recurrent_dropout=dropout))
    #model.add(LSTM(NEURONS_LSTM, return_sequences=True, dropout=dropout))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout1))

    # model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(100, 11)))
    # model.add(Bidirectional(LSTM(50)))
    # # model.add(Dense(100, activation='relu', input_shape=X.shape))
    # # model.add(Dense(100, activation='relu'))

    # Output layer
    model.add(Dense(11, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Model and parameter settings
# Output model
Y = pd.get_dummies(trainingBig["y_trainBig"])

# for i in [10, 70]:
X = tokenizer.texts_to_sequences(trainingBig["x_trainBig"])
X = pad_sequences(X, maxlen=200)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=12)

# NEURONS_LSTM = round((2/3)*(i+11))
epochs = 100
batch_size = 32

accuracy = []

for i in range(0,5):
    model = RNN(33, 141, 0.4, 0.1)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    acc = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(acc[0], acc[1]))
    accuracy.append(acc[1])

print(accuracy)
print(np.mean(accuracy))

# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

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

# Input confusion matrix
predicted = model.predict_classes(X_test)
Y_test = Y_test[Y_test==1].stack().reset_index().drop(0, 1)

# Confusion matrix, does not work correctly yet
conf_mat = confusion_matrix(Y_test["level_1"], predicted)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.newHashtag.values, yticklabels=category_id_df.newHashtag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()