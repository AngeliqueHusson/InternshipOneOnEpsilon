"""
    Adding video titles to caption text, in order to include in feature selection

    @authors Angelique Husson & Nikki Leijnse
"""

import nltk
import re
import os
import string
import json
import pandas as pd
from nltk.tokenize import word_tokenize

findtxt = re.compile(r'[0-9a-zA-Z]+\.json')
findtxt.findall(r'new.json*****new.json')

# Data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/video_info'
os.chdir(directory)
filelist = os.listdir(directory)
print(len(filelist))

# Initialize empty dataframe
matrixdf = pd.DataFrame(columns=['youtubeVideoId', 'title'])
ntitle = 3  # How many times do you want to add the title?

for i in filelist:
    with open(i, errors='ignore') as file:
        data = json.load(file)
        for j in data['items']:
            title = j['snippet']['title']
            title = str(title)

            # Removing the .json characters of the file string
            id = str(i)
            id = id[:-5]

            # Adding title
            title1 = title+' '
            # Put more weight on titles, in this case they weigh 3 times more than the text.
            matrixdf2 = matrixdf.append({'youtubeVideoId': id, 'title': ntitle*title}, ignore_index=True)
            matrixdf = matrixdf2


directory2 = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption'
os.chdir(directory2)
filelist = os.listdir(directory2)

nrow = len(matrixdf)
print(nrow)  # Less than 1504

for i in filelist:
    with open(i,errors='ignore') as file:
        textString = file.read().replace('/n', '')
        words = word_tokenize(textString)

        # Removing the .txt characters of the file string
        id = str(i)
        id = id[:-4]

        # Check for matching video id
        for k in range(0, nrow):
            if id == matrixdf.loc[:, 'youtubeVideoId'].values[k]:
                x = matrixdf.loc[k, 'title']
                x = str(x)
                words.append(x)  # Adding the titles to the captions

    # write into new file
    new_path = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption title'
    new_text = ' '.join(words)
    file_name = os.path.join(new_path, i)
    f = open(file_name, 'w')
    f.write(new_text)
    f.close()


