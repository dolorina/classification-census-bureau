'''
Script to train machine learning model.

Author: Marina Dolokov
Date: January 2022
'''

from os import sep
from sklearn.model_selection import train_test_split
import pandas as pd
from data import process_data
from model import train_model, inference, load_model, compute_model_metrics, metrics_on_fixed_features
from numpy import savetxt,array

# load in the data
data = pd.read_csv('../data/census_preprocessed.csv')
data = data.drop(*["Unnamed: 0"], axis=1)
data = data.drop(*["fnlgt"], axis=1)
data = data.drop(*["education-num"], axis=1)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_feat = [
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
    train, categorical_features=cat_feat, label="salary", training=True)

X_test, y_test, _, _ = process_data(
    X=test, categorical_features=cat_feat, label="salary", training=False, 
    encoder=encoder, lb=lb)

# training and saving the model 
model = train_model(X_train, y_train) 
# model = load_model() 
preds = inference(model, X_test)
precision, recall, f1, accuracy = compute_model_metrics(y_test, preds)

print("Metrics:")
print("    prec={}, rec={}, f1={}, acc={}".format(round(precision, 3), round(recall, 3), round(f1, 3), round(accuracy, 3)))
print()

print('Metrics in salary')
metrics_on_salary = metrics_on_fixed_features(model, test, cat_feat, encoder, lb, feature="salary")
print('Type', type(metrics_on_salary))
print(metrics_on_salary)
print()

print('Metrics in sex')
metrics_on_sex = metrics_on_fixed_features(model, test, cat_feat, encoder, lb, feature="sex")
print(metrics_on_sex)
print()

print('Metrics in education')
metrics_on_education = metrics_on_fixed_features(model, test, cat_feat, encoder, lb, feature="education")
round(metrics_on_education, 3).to_csv("./slice_output.txt", header=None, index=None, sep=' ', mode='a')
print(metrics_on_education)
print()

print('Metrics in workclass')
metrics_on_workclass = metrics_on_fixed_features(model, test, cat_feat, encoder, lb, feature="workclass")
print(metrics_on_workclass)