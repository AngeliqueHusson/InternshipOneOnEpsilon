"""
    This file joins the file with the hashtags and the file with the text using the videoId
    It also initializes a training, validation and test set.

    Run this file after changing the hashtags in 'Change Hashtags.py'

    @authors Angelique Husson & Nikki Leijnse
"""

import os
import pandas as pd
from nltk import word_tokenize
from sklearn.model_selection import train_test_split

directory = os.getcwd()
directory = "C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/"
os.chdir(directory)

data = pd.read_csv("Data/newHashtags.csv")
print(data.columns.ravel())  # In order to find titles of columns
new_data = data[['youtubeVideoId', 'newHashtag']]

# Choose your own directory here
directory2 = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after stemming'
os.chdir(directory2)
filelist = os.listdir(directory2)

# Initialize empty dataframe
matrixdf = pd.DataFrame(columns=['youtubeVideoId', 'text'])

for i in filelist:
    with open(i, encoding='gb18030', errors='ignore') as file:
        textString = file.read().replace('/n', '')
        words = word_tokenize(textString)

        # Removing the .txt characters of the file string
        j = str(i)
        j = j[:-4]

        matrixdf2 = matrixdf.append({'youtubeVideoId': j, 'text': words}, ignore_index=True)
        matrixdf = matrixdf2


# Testing which videos are not merged
new = pd.concat([new_data, matrixdf2], keys='youtubeVideoId')
print(new.count())

# Merge data by the video ID
fulldf = pd.merge(new_data, matrixdf2, on='youtubeVideoId', validate='one_to_one', indicator=True)
print(fulldf)
print(new_data.count())
print(matrixdf2.count())
print(fulldf.count())  # Final dataset has less values?

fulldf['y_id'] = fulldf['newHashtag'].factorize()[0]
category_id_df = fulldf[['newHashtag', 'y_id']].drop_duplicates().sort_values('y_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['y_id', 'newHashtag']].values)
print(fulldf.head())

# Save file as csv file
os.chdir(directory)
category_id_df.to_csv('Data/category_id_df.csv')
fulldf.to_csv('Data/HashtagText.csv')  # Complete csv file with all information
new.to_csv('Data/identifying_videos_without_text.csv')

# Splitting and saving training, validation and test set data
x_trainBig, x_test, y_trainBig, y_test = train_test_split(fulldf['text'], fulldf['y_id'], test_size=0.2, random_state=60)
x_train, x_val, y_train, y_val = train_test_split(x_trainBig, y_trainBig, test_size=0.2, random_state=12)

# Big training set: training and validation set combined
trainingbig = pd.DataFrame({'x_trainBig': x_trainBig, 'y_trainBig': y_trainBig})
print(trainingbig.count())
trainingbig.to_csv('Data/trainingbig.csv')

# Training set
trainingdf = pd.DataFrame({'x_train': x_train, 'y_train': y_train})
print(trainingdf.count())
trainingdf.to_csv('Data/training.csv')

# Validation set
validationdf = pd.DataFrame({'x_val': x_val, 'y_val': y_val})
print(validationdf.count())
validationdf.to_csv('Data/validation.csv')

# Testing set
testingdf = pd.DataFrame({'x_test': x_test, 'y_test': y_test})
print(testingdf.count())
testingdf.to_csv('Data/testing.csv')
