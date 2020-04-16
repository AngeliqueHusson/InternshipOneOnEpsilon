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
data_df = pd.read_excel('Hashtags.xlsx', sheet_name='Blad1')
# Write initial hashtags into file
df_new.to_csv('InitialHashtags.csv')

# Initialization
columns = data_df.columns.ravel()
nrow = data_df.count()
ncol = len(columns)

# Only use last hashtag
df_new['chosenHashtag'] = df_new['hashTags'].str[-1]
df_new['newHashtag'] = np.nan  # New hashtag

for j in range(0, ncol):  # Columns
    for k in range(0, nrow[j]):  # Rows in each column
        x = data_df.loc[:, columns[j]].values[k]
        x = str(x)

        # If a hashtag in our long list matches one in the our excel file,
        # it sets this hashtag to the corresponding column in our excel file
        df_new['newHashtag'][df_new['chosenHashtag'] == y] = columns[j]

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