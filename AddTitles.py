
# Step 1:
# Obtain tiles

import nltk
import re
import os
import string
import json
import pandas as pd
from nltk.tokenize import word_tokenize

# Porter stemming method

# I do not know if this is important
findtxt = re.compile(r'[0-9a-zA-Z]+\.json')
findtxt.findall(r'new.json*****new.json')

# Data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/video_info'
os.chdir(directory)
filelist = os.listdir(directory)
print(len(filelist))

# Initialize empty dataframe
matrixdf = pd.DataFrame(columns=['youtubeVideoId', 'title'])

for i in filelist:
    with open(i, errors='ignore') as file:
        data = json.load(file)
        for j in data['items']:
            title1 = j['snippet']['title']

            textString = title1.replace('/n', '')
            title = word_tokenize(textString)

            # Removing the .txt characters of the file string
            id = str(i)
            id = id[:-5]

            matrixdf2 = matrixdf.append({'youtubeVideoId': id, 'title': title+title+title}, ignore_index=True)
            matrixdf = matrixdf2


directory2 = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption'
os.chdir(directory2)
filelist = os.listdir(directory2)

nrow = len(matrixdf)
print(nrow)

for i in filelist:
    with open(i,errors='ignore') as file:
        textString = file.read().replace('/n', '')
        words = word_tokenize(textString)

        # Removing the .txt characters of the file string
        id = str(i)
        id = id[:-4]

        for k in range(0, nrow):
            if id == matrixdf.loc[:, 0].values[k]:
                k2 = matrixdf.loc[k,1]
                k3 = str(k2)
                x = matrixdf.loc[k3, 'title']
                print(x)
                words.append(x)



    # write into new file
    new_path = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption title'
    new_text = ' '.join(words)
    file_name = os.path.join(new_path, i)
    f = open(file_name, 'w')
    f.write(new_text)
    f.close()


