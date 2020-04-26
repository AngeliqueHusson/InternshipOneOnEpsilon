"""
    Retrieval and cleaning of the video titles, in order to include titles in feature selection

    @authors Angelique Husson & Nikki Leijnse
"""

import re
import os
import string
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words

findtxt = re.compile(r'[0-9a-zA-Z]+\.json')
findtxt.findall(r'new.json*****new.json')

# Data import
directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data/video_info'
os.chdir(directory)
filelist = os.listdir(directory)
print(len(filelist))

# Initialize empty dataframe
matrixdf = pd.DataFrame(columns=['youtubeVideoId', 'title'])

# Implementation of dictionary
My_dict = {}
for i in words.words():
    My_dict[i] = i

# Retrieve titles and video ids
for i in filelist:
    with open(i, errors='ignore') as file:
        data = json.load(file)
        for j in data['items']:
            title = j['snippet']['title']
            textString = str(title)

            ### The cleaning code below has been retrieved from the Github repository of Hui Dong:
            ### https://github.com/donghui435/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles-

            # Eliminate the punctuation in form of characters
            textString = [char for char in textString if char not in string.punctuation]
            textString = ''.join(textString)
            textString = textString.lower()

            # Tokenize
            textString_token = word_tokenize(textString)

            # Stopwords elimination 1st time
            textString_stop = [word for word in textString_token if word not in stopwords.words('english')]

            # Spelling checking and dividing two connected words
            list1 = []
            for word in textString_stop:
                if word in My_dict:
                    list1.append(word)
                else:
                    list2 = []
                    count_j = 0
                    for j in range(1, len(word)):
                        if word[:j] in My_dict and word[j:] in My_dict:
                            list2.append(word[:j])
                            list2.append(word[j:])
                            count_j += 1
                    if count_j == 1:
                        list1.extend(list2)

            # Stopwords elimination 2nd time
            clean_title = [word for word in list1 if word.lower() not in stopwords.words('english')]
            clean_title = ' '.join([str(w) for w in clean_title])

            # Removing the .json characters of the file string
            id = str(i)
            id = id[:-5]

            # Connect video id and title in matrix
            matrixdf2 = matrixdf.append({'youtubeVideoId': id, 'title': clean_title}, ignore_index=True)
            matrixdf = matrixdf2

# Save as csv file
directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
os.chdir(directory)
matrixdf.to_csv('titles.csv', index=False)