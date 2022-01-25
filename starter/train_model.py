'''
Script to train machine learning model.

Author: Marina Dolokov
Date: January 2022
'''

from os import sep
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.insert(1, './ml') # change sys.path for function import
from data import process_data
from model import train_model, inference, compute_model_metrics

# load in the data
data = pd.read_csv('../data/census_preprocessed.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# processing train and test data with process_data function
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

X_test, y_test, encoder, lb = process_data(
    X=test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

# training and saving the model 
model = train_model(X_train, y_train)
preds = inference(model, X_test)
accuracy, precision, recall, fbeta = compute_model_metrics(y_test, preds)
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)
print('F1', fbeta)