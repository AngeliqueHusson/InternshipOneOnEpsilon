# InternshipOneOnEpsilon
 Text classification for videos from the start-up company One on Epsilon.
 This repository is a continuation of the existing repository:
 https://github.com/donghui435/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles-

## Add Titles
With this method we select the titles as well as the captions of the videos and combine them in one new file. As we want to put more weight on words that exist in the title during the feature selection process, the words of the title are added three times in this case to the new file. One could also choose to change this weight. 

## Data cleaning
This method is part of the following repository:  
https://github.com/donghui435/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles-

## Stemming
With this method we convert all word forms to the canonical (root) form. For this we use Porter's method.

## Change Hashtags
The original data set contained many overlapping and related hashtag. As the number of videos related to an individual hashtag is small, the overlapping and related hashtags are combined and given a new umbrella hashtag.

## Splitting data into training and test set
With this method we combine the textfiles obtained with "add titles" with their corresponding new hashtag retrieved from "change hashtags" into one data file. After that, we split the data in a training and a test set which are all used with the classifiers described below.


