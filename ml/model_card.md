
# Salary prediction based on census bureau

## Model Details

The model was created by M. Dolokov. It is a multi layer perceptron classifier from scikit-learn 1.0.2. 

## Intended Use

This model should be used to predict whether a person gains more or less than 50 k salary based on information about the age, the sex, the education and other information about the person. The user could be an insurance company or a financial advice company. 

## Metrics

The model was evaluated using the precision metric. The value of precision is prec=0.708.  
Also recall, f1 score and accuracy were calculated. The values are rec=0.651, f1=0.678, acc=0.849. 

## Data

The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income). It was donated by Ronny Kohavi and Barry Becker. Extraction was done by Barry Becker from the 1994 Census database. 

Prediction task is to determine whether a person makes over 50K a year.

The original data set has 15 variables and 32561 observations. 

To use the data for training a One Hot Encoder from scikit-learn was used on the features and a label binarizer also from scikit-learn was used on the labels.

A 80-20 split was used to break into a train and test set while no stratification was done. 

### Training data 

The training data set has 26048 observations (80% of original data). The categorical features were passed through a One Hot Encoder to make them machine readable. 

### Evaluation data 

The testing data set has 6513 observations (20% of orignal data). In the same way as within the training data, categorical features were transformed by a One Hot Encoder. 

## Ethical Considerations, Caveats and Recommendations

When using this dataset to train machine learning models, this should be done with care. The dataset contains sensible information about race, sex, maritial status, salary and much more. Even though there maight be correlations between those features, what can lead to well performing models, it does not reflect how decision should be made. For instance, the race of a person should not influence the salary of that person. It is not recommended to make decision about salary of new employees based models that were trained on the information provided by this dataset. A dataset can have a bias that reflects discrimination. To make a machine learning models fair and not discriminative, information about race, sex etc. should be excluded from the training and testing data. Such information can be used as additional information and for statistics, not for model training. 

