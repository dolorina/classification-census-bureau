from model import compute_model_metrics, inference
from data import process_data
import pandas as pd
from sklearn.model.selection import train_test_split
import sys
sys.path.insert(1, '../starter') # change sys.path for function import
from data import train_model

def load_data():
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
    return train, test, cat_features

# processing train and test data with process_data function
def test_compute_model_metrics():
    train, test, cat_feat = load_data()
    X_train, y_train, _, _ = process_data(train, categorical_features=cat_feat, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_feat, label="salary", training=True)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    acc, prec, rec, f1 = compute_model_metrics(y_test, preds)
    assert(acc >= 0.5)
    assert(prec >= 0.5)
    assert(rec >= 0.5)
    assert(f1 >= 0.5)

def test_process_data():
    train, test, cat_feat = load_data()
    X, y, _, _ =process_data(train, categorical_features=cat_feat, label="salary", training=True)
    assert(X.shape[0]==y.shape[0])
    assert(type(X)==type(y)=="np.array")

def test_inference():
    train, test, cat_feat = load_data()
    X_train, y_train, _, _ = process_data(train, categorical_features=cat_feat, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_feat, label="salary", training=True)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert(preds==model.predict(X_test))
    assert(preds.shape==y_test.shape)
    assert(type(preds)=="np.array")