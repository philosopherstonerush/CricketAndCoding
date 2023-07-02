![](https://github.com/philosopherstonerush/CricketAndCoding/assets/77642143/4b033902-6e00-4dc4-9af2-577927d5050b)

# Cricket and Coding
The following project is what I used to participate and eventually win in the national level machine learning/data science hackathon called "Cricket and Coding" conducted by IIT madras in the year 2023.

# Overview
The model file takes in some of the columns specified in the dataset present inside the assets folder and then trains a model according to it.

The main file then takes various inputs from the test_file.csv and then returns the predicted score after 6 overs given an inning. The output appears in submission.csv

# What algorithm is used? 
Random forest regression, provided by the scikit-learn package is used to find the score of any given team. 

# Performance
The model successfully predicted scores for over 100 innings with an average error of 13 points per inning.
