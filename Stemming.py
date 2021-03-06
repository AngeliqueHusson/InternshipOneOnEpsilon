"""
Stemming method and further data cleaning
Run this file after 'Data Cleaning.py'
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
import re
import os

# Porter stemming method
ps = PorterStemmer()

findtxt = re.compile(r'[0-9a-zA-Z]+\.txt')
findtxt.findall(r'new.txt*****new.txt')

# Data import, change to your own directory
# directory = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data/Caption title'
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption title'
os.chdir(directory)
filelist = os.listdir(directory)
print(filelist)

for i in filelist:
    with open(i, errors='ignore') as file:
        textString = file.read().replace('/n','')
        words = word_tokenize(textString)
        list = []

        for w in words:
           x = ps.stem(w)  # Stemming method
           list.append(x)

    # write into new file (Change to your own path)
    # new_path = 'C:/Users/Nikki/Desktop/Internship AM/Input data classification/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles--master/Data/Caption after stemming/'
    new_path = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after stemming/'
    new_text = ' '.join(list)
    file_name = os.path.join(new_path, i)
    f = open(file_name, 'w')
    f.write(new_text)
    f.close()
