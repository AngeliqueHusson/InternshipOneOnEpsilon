"""
    Adding video titles to caption text, in order to include in feature selection

    @authors Angelique Husson & Nikki Leijnse
"""

import os
import pandas as pd
from nltk.tokenize import word_tokenize

# Set percentage that the title should account for in feature selection
percentage = 20

# Retrieve file with titles (Change to your own directory)
#directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data'
os.chdir(directory)

df = pd.read_csv("titles.csv")
nrow = len(df)

# Combining titles and captions
#directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data/Caption after clean'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after clean'
os.chdir(directory)
filelist = os.listdir(directory)

for i in filelist:
    with open(i,errors='ignore') as file:
        textString = file.read().replace('/n', '')
        qw = len(textString.split())
        words = word_tokenize(textString)

        tp = percentage

        # Removing the .txt characters of the file string
        id = str(i)
        id = id[:-4]

        # Check for matching video id
        for k in range(0, nrow):
            if id == df.loc[:, 'youtubeVideoId'].values[k]:
                x = df.loc[k, 'title']
                title = str(x)

                qx = len(title.split())

                tl = tp*qw/(100-tp)
                m = round(tl/qx)

                for j in range(1,m):
                    words.append(title)  # Adding the titles to the captions

    # Write into new file (Change to your own directory)
    # new_path = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data/Caption title'
    new_path = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption title'
    new_text = ' '.join(words)
    file_name = os.path.join(new_path, i)
    f = open(file_name, 'w')
    f.write(new_text)
    f.close()
