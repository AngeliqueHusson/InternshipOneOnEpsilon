# InternshipOneOnEpsilon
Text classification for videos from the start-up company One on Epsilon.
This repository is a continuation of the existing repository:
https://github.com/donghui435/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles-
 
## Title retrieval and cleaning
Using the aforementioned repository we only retrieve the captions of the videos. In the text classification process we want to include the titles as well in the feature space. Therefore, this method is used to retrieve the titles of the different videos. In this method we also "clean" the titles using the data cleaning method from the aforementioned repository.

## Add titles
With this method we select the titles as well as the captions of the videos and combine them in one new file. As we want to put more weight on words that exist in the title during the feature selection process, we choose a percentage of the total amount of words that should consist out of words from the title. Initially this is set equal to 10%.

## Stemming
With this method we convert all word forms to the canonical (root) form. For this we use Porter's method.

## Change hashtags
The original data set contained many overlapping and related hashtags. As the number of videos related to an individual hashtag is small, the overlapping and related hashtags are combined and given a new umbrella hashtag.

## Joining and splitting data
With this method we combine the textfiles obtained with "add titles" with their corresponding new hashtag retrieved from "change hashtags" into one data file. After that, we split the data in a training and a test set which are all used with the classifiers described below.


