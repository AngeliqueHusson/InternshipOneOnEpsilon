# Changing the Hashtags to the new hashtags
import os
import urllib.request, json
import pandas as pd

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
df_new['newHashtags'] = df_new['hashTags'].str[-1]

for j in range(0, ncol):  # Columns
    for k in range(0, nrow[j]):  # Rows in each column
        x = data_df.loc[:, columns[j]].values[k]
        x = str(x)
        y = str('#'+x)

        # If a hashtag in our long list matches one in the our excel file,
        # it sets this hashtag to the corresponding column in our excel file
        df_new['newHashtags'][df_new['newHashtags'] == y] = columns[j]


print('This is the matrix:')
print(df_new)
print(df_new.count())



df_new.to_csv('newHashtag3.csv')
df_new[['youtubeVideoId','newHashtags']].to_csv('newHashtags4.csv')