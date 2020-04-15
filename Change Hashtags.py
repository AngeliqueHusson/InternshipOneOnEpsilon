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

columns = data_df.columns.ravel()
nrow = data_df.count()
ncol = len(columns)
matrixdf = pd.DataFrame(columns=['youtubeVideoId', 'newHashtags'])

found = True

for i in range(0, len(df_new['hashTags'])):  # Hashtags from internet list
    if found == False:
        print(df_new['youtubeVideoId'].values[i])
    found = False
    print(i)
    for j in range(0, ncol):  # Columns
        for k in range(0, nrow[j]):  # Rows in each column
            x = data_df.loc[:, columns[j]].values[k]
            x = str(x)
            y = str('#'+x)

            # Only takes last value if there are multiple hashtags
            string = df_new['hashTags'].str[-1].values[i]
            string = str(string)

            if string == y:
                id = df_new['youtubeVideoId'].values[i]
                c = columns[j]
                matrixdf2 = matrixdf.append({'youtubeVideoId': id, 'newHashtags': c}, ignore_index=True)
                matrixdf = matrixdf2
                found = True


print('This is the matrix:')
print(matrixdf)
print(matrixdf.count())

matrixdf.to_csv('newHashtag.csv')