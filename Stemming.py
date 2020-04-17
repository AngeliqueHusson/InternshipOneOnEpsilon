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

# Data import
directory = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after clean'
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

    # write into new file
    new_path = 'C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/Caption after clean 2/'
    new_text = ' '.join(list)
    file_name = os.path.join(new_path, i)
    f = open(file_name, 'w')
    f.write(new_text)
    f.close()
