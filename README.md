project proposal
================
# Abstract
In recent years, for people of all ages and all places, PC games have become an irreplaceable part of their daily lives. Among the most famous digital gaming distribution platforms, Steam not only attracts tons of loyal customers, but also has a highly admirable praise rate of their games. In this project, our team intends to find out the underlying factors and the relationship between the praise rate with them by implementing several different supervised learning algorithms on an open data source provided by Steam. By the end of the project phase, we would build a model that is able to predict the praise rate for a new game.
# Introduction
As is well known，the Steam platform is one of the biggest digital distribution platforms for PC games. By 2017, users purchased games through the Steam store for a total of approximately US $ 4.3 billion, accounting for at least 18% of PC game sales worldwide [1]. By 2019, the Steam service has more than 34,000 games and more than 95 million monthly active users [1]. In this context, it is worth studying the underlying factors behind the popularity, and discovering the relationships between the high praise with them.<br>
<br>
In this project, we will measure the influence coefficient of each feature on the praise rate of different games by analyzing the open data source. Then, implementing several supervised learning algorithms to build a model to predict the score of a new game.<br>
<br>
More specifically, the proportion of positive comments in the total number of comments will be used as a standard for the positive rate of a game. Apart from that, the following factors will also be considered: release date, English, developer, publisher, platforms, required age, categories, genres, Steamspy tags, achievements, average playtime, median playtime, owners, price. By measuring the correlation of these factors,  the praise rate of a game will be eventually predicted.<br>
<br>

For the smooth progress of this project, we will carry out the following related work. Firstly, we will check the relevant papers and online materials to confirm the reliability of the research direction, find the relevant data sets and analyze which data in the data sets can be used in the project. Secondly, select the applicable data analysis technologies and supervised learning algorithms to build the model. After training and testing the model, it can be used to predict the praise rate of a game. All the above work requires the fair division and cooperation of all the team members. In order to reach that, we plan to discuss the project progress and solve the difficulties encountered on a weekly basis.<br>
# Dataset
The dataset we will use in this project is retrieved from https://www.kaggle.com[2], which is composed of more than 27000 games’ information gathered from Steam store and SteamSpy APIs around May 2019. <br>

As indicated in the introduction, the selected features are released date, support english, developer, publisher, platforms, required age and so on. The dataset has high dimension regarding to the number of features, and the scale of data available for model training is relative large(27000 rows).
# Technologies
For the whole project, we decided to mainly use two libraries, Spark and sciki-learn.
## Spark
Spark offers various interfaces which can be used to read and preprocessing our data.             We plan to use pyspark.sql.dataframe to retrieve our data from csv file to dataframe, and then apply some operations to form our data for the input of our model.
## Sciki-learn
As a free software machine learning library for python, sklearn features a various regression algorithms, including randomforest, decision trees which will be used for our project.
# Algorithms
After basic analysis, brainstorming and background research on the scope of our project and the characteristics of dataset, we decided to apply the following techniques or algorithms in our project, to get an accurate and efficient model:
## Linear Regression
Linear regression is a common Statistical Data Analysis technique.  It is used to determine the extent to which there is a linear relationship between a dependent variable and one or more independent variables. There are two types of linear regression, simple linear regression and multiple linear regression. We observed that there are more than 5 features is directly proportional to the results(rate of our good feedback). Therefore, we plan to try multiple linear regression model first to do the prediction. 
## KNN
As far as we know, KNN can be used for both classification and regression problems. The algorithm uses ‘feature similarity’ to predict values of any new data points. This means that the new point is assigned a value based on how closely it resembles the points in the training set. In our project, we plan to classify the games with the features into several categories and do the regression.

## Random Forest
As the dimension of our data is low, random forest is more proper to choose. Besides, it maintains high accuracy.  After we apply Onehot encoding and k fold cross-validation, we would generate the random forest model with sklearn. And then we may feed in some Numpy arrays for training, ask it to make a prediction, evaluate the predictions, and see that they are reasonable. For checking if our predicting value is reasonable or not, we can pick up any tree we have generated and look at the feature importances of our explanatory variables.
  
## One-hot encoding
Since 80% of features selected for this project are categorical and discrete, it is not a good idea to simply give an integer to represent each one of the categorical features, and feed them directly to the training phase. Because without any pre-processing, the output model will ‘think’ that the higher/lower the categorical value is, the better/worse the category is. However in fact, every category should be treated equally.<br>
One-hot encoding is the solution for this problem, by representing each features with binary vectors. In such case, the model will not have preference to a specific data point because it has better value representing its category.
## K fold cross-validation
To assure the accuracy of our model and prevent under/over fitting, the evaluation of the performance of the model is required. To achieve that, we chose k folder cross-validation to re-sampling our dataset. The exact value for k will be decided during the implementation of this project.


# Reference
[1] Steam (service), last edited on 8 February 2020, https://en.wikipedia.org/wiki/Steam_(service)<br>
[2] Steam Store Games (Clean dataset). (2020). Retrieved 11 February 2020, from https://www.kaggle.com/nikdavis/steam-store-games<br>

[3] What is One Hot Encoding? Why And When do you have to use it?. (2020). Retrieved 11 February 2020, from 
https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f<br>
