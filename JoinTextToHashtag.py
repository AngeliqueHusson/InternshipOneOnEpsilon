import re
import os
import pandas as pd
import numpy as np
from nltk import word_tokenize

directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/'
os.chdir(directory)
data = pd.read_csv("newHashtags4.csv")
print(data.columns.ravel())  # In order to find titles of columns
new_data = data[['youtubeVideoId', 'newHashtags']]

directory2 = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after clean 2'
os.chdir(directory2)
filelist = os.listdir(directory2)

# Initialize empty dataframe
matrixdf = pd.DataFrame(columns=['youtubeVideoId', 'text'])

for i in filelist:
    with open(i, encoding='gb18030', errors='ignore') as file:
        textString = file.read().replace('/n','')
        words = word_tokenize(textString)

        # Removing the .txt characters of the file string
        j = str(i)
        j = j[:-4]

        matrixdf2 = matrixdf.append({'youtubeVideoId': j, 'text': words}, ignore_index=True)
        matrixdf = matrixdf2


# Merge data by the video ID
fulldf = pd.merge(new_data, matrixdf2, on='youtubeVideoId')
print(fulldf)
print(new_data.count())
print(matrixdf2.count())
print(fulldf.count())  # Final dataset has less values?

# Save file as csv file
os.chdir(directory)
fulldf.to_csv('HashtagText.csv')
