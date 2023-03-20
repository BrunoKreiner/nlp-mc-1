# nlp-mc-1

## Objective

As part of my 5th semester project, 49985 songs have been classified with a list of genres pertaining to the artist of the particular song. For each song and their artist(s) there exists some meta data. The objective is to classify the correct genre(s) of each song based on their lyrics since crawling the lyrics of those songs was a big task in the project. More meta data exists for the particular song (Spotify API and additional data from the project) that can be used for improving classification. This meta data will not be included in the classification in the first step. 

## Preprocessing

The lyrics have already been cleaned from non alphanumeric characters. The preprocessing in [preprocess.ipynb](./notebooks/preprocess.ipynb) is aimed towards simplifying genre classification by reducing the total number of classes (N = 10).