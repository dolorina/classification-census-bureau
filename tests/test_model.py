from ml.model import compute_model_metrics, inference, train_model
from ml.data import process_data

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

def load_data():
    # load in the data
    data = pd.read_csv('../data/census_preprocessed.csv')
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
    return train, test, cat_feat

def test_process_data():
    train, test, cat_feat = load_data()
    X, y, _, _ = process_data(train, categorical_features=cat_feat, label="salary", training=True)
    assert(X.shape[0]==y.shape[0])
    assert(type(X) == type(y) == np.ndarray)

def test_compute_model_metrics():
    train, test, cat_feat = load_data()
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_feat, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_feat, label="salary", training=False, encoder=encoder, lb=lb)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    acc, prec, rec, f1 = compute_model_metrics(y_test, preds)
    assert(isinstance(acc, float))
    assert(isinstance(prec, float))
    assert(isinstance(rec, float))
    assert(isinstance(f1, float))
    # assert(acc >= 0.5)
    # assert(prec >= 0.5)
    # assert(rec >= 0.5)
    # assert(rec >= 0.5)

def test_inference():
    train, test, cat_feat = load_data()
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_feat, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_feat, label="salary", training=False, encoder=encoder, lb=lb)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert(preds.all()==model.predict(X_test).all())
    assert(preds.shape==y_test.shape)
    assert(isinstance(preds, np.ndarray))