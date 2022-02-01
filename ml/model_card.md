
# Salary orediction based on census bureau

## Model Details

The model was created by M. Dolokov. It is a multi layer perceptron classifier from scikit-learn 1.0.2. 

## Intended Use

This model should be used to predict whether a person gains more or less than 50 k salary based on information about the age, the sex, the education and other information about the person. The user could be an insurance company or a financial advice company. 

## Metrics

The model was evaluated using the precision metric. The value of precision is prec=0.803.  

## Data

The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income). It was donated by Ronny Kohavi and Barry Becker. Extraction was done by Barry Becker from the 1994 Census database. 

Prediction task is to determine whether a person makes over 50K a year.

The original data set has 15 variables and 32561 observations. A 80-20 split was used to break into a train and test set while no stratification was done. 

To use the data for training a One Hot Encoder from scikit-learn was used on the features and a label binarizer also from scikit-learn was used on the labels.