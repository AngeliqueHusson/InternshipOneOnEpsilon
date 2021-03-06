# InternshipOneOnEpsilon
Text classification for videos from the start-up company One on Epsilon.
This repository is a continuation of the existing repository:
https://github.com/donghui435/YouTube-video-info-download-including-title-channel-automatically-generated-subtitles-
 
## Data Cleaning
This file originates from the before mentioned repository. It takes as input the data files in the data directory 'Caption'. This file cleans the video captions data by removing stop words and dividing connected words. Running this file saves the cleaned video captions in the directory 'Caption after cleaning'. If the data in this directory is already downloaded or obtained, there is no need to run this file. 
 
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

## Classification methods
After running the files in the previous steps and obtaining the training, test and validation datasets the user can run several classification methods on this data. The methods Naive Bayes, Logistic Regression, Support vector machine, Random Forest and a Recurrent Neural Network can be run in the corresponding python files. The files with '_final' at the end of the file are used on the test data.

## Correlations
In order to see the most correlated words for each hashtag or label, the correlations file can be run. This file outputs for each hashtag the two most important words. 

## Website
The resulting classification algorithms are saved as pickled files in the directory 'Data/Webpage'. These are used in order to create a website that shows the functionalities of these classification algorithms. There two python files 'Website.py' and 'WebsiteURL.py' can use the implemented algorithms to make predictions of users input. The static version of this website can be seen at: https://angeliquehusson.github.io/MathClassification/ .
The website that has working interactive components is hosted on a server and can temporarily be seen at MathClassification.win.tue.nl
The code used to create this website can be seen in repository: https://github.com/AngeliqueHusson/MathClassificationWebsite .

## Learning curves and sample size prediction
In order to determine the required sample size for a certain accuracy, one could create the learning curve of the classifier. This means that we plot the training accuracy against the validation accuracy for different sample sizes. Once we have obtained this learning curve, we can fit a power function to the validation curve and predict the number of samples needed to obtain a certain accuracy. 


