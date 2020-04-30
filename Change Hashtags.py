"""
 Changing initial 760 Hashtags to the newly created hashtags
 Run this file after the Data cleaning steps

 @authors Angelique Husson & Nikki Leijnse
"""

# Changing the Hashtags to the new hashtags
import os
import urllib.request, json
import pandas as pd
import numpy as np

with urllib.request.urlopen("https://s3.amazonaws.com/oneonepsilon-database/database.json") as url:
    database = json.loads(url.read().decode())

videolist = database['videos']
df = pd.DataFrame(videolist)
df_new = df[['youtubeVideoId', 'hashTags']]
#print(df_new[:50])

# Read excel
os.chdir('C:/Users/s157165/Documents/Jaar 5 2019-2020 Master/Internship Australia/InternshipOneOnEpsilon/Data/')
data_df = pd.read_excel('Hashtags.xlsx', sheet_name='Sheet2')
# Write initial hashtags into file
df_new.to_csv('InitialHashtags.csv')

# Initialization
columns = data_df.columns.ravel()
nrow = data_df.count()
ncol = len(columns)

# Only use last hashtag
new = df_new.hashTags.apply(pd.Series).add_prefix('hash_')
nc = new.shape[1]
print(nc)

for m in range(0,nc):
    i = str(m)
    new[i] = np.nan
# new1 = new.append(np.nan)

print(new)

for i in range(0,nc):
    for j in range(0, ncol):  # Columns
        for k in range(0, nrow[j]):  # Rows in each column
            x = data_df.loc[:, columns[j]].values[k]
            x = str(x)

            # If a hashtag in our long list matches one in the our excel file,
            # it sets this hashtag to the corresponding column in our excel file
            # print(new.iloc[:,i])
            m = str(i)
            new[m][new['hash_'+str(i)] == x] = columns[j]

print(new)

new1 = new.iloc[:,nc:2*nc]
new2 = new1.mode(axis=1).iloc[:,0]
print(new2)

new["newHashtag"] = new2
new.to_csv('tijdelijk.csv')

df_new['newHashtag'] = new2

print('Number of rows')
print(df_new.count())

# Drop rows that are not matched
df_new = df_new.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

print(df_new)
print("Number of rows after removing unmatched hashtags")
print(df_new.count())

# Saving to files
df_new.to_csv('newHashtagsFull.csv')  # File with all the information
df_new[['youtubeVideoId','newHashtag']].to_csv('newHashtags.csv')  # File with info needed to continue