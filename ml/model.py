'''
Script that contains functions which train a model, compute predictions with this model and evaluate the modelprocesses data for machine learning 

Author: Marina Dolokov
Date: Februar 2022
'''

from __future__ import absolute_import

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ml

from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from pandas import DataFrame 
from numpy import array, append

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    parameter_space = {
        'hidden_layer_sizes': [(200, 100, 10,), (100, 50, 10,), (100,), (20,), (10,)],
        'activation': ['relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.001, 0.0001],
        'learning_rate': ['constant','adaptive'],
    }
    # parameter_space = {
    #     'hidden_layer_sizes': [(10,)],
    #     'activation': ['relu'],
    #     'solver': ['adam'],
    #     'alpha': [0.001],
    #     'learning_rate': ['constant'],
    # }
    print("Find best hyperparameter setting.")
    print()
    
    model = MLPClassifier(max_iter=5000) 
    model = GridSearchCV(model, parameter_space, n_jobs=-1, cv=5)
    model.fit(X_train, y_train)    

    print("Best parameters found:\n", model.best_params_)
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params, in zip(means, stds, model.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))
    pickle.dump(model, open("./mlp_classifier.sav", 'wb')) 
    return model

def load_model(name="./mlp_classifier.sav"):
    '''
    Loads a trained and saved machine learning model and return it. 

    Returns
    -------
    model 
        Saved and trained machine learning model
    '''
    model = pickle.load(open(name, "rb"))
    return model


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model 
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, F1 and accuracy.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    accuracy : float 
    precision : float
    recall : float
    fbeta : float
    """
    accuracy = accuracy_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    # print(classification_report(y, preds))
    return precision, recall, fbeta, accuracy

def metrics_on_fixed_features(model, test, cat_feat, encoder, lb, feature="education", label="salary"):
    '''
    Validates trained ml model on sclices of data
    Inputs
    ------
    model :  (trained ml model)
    test : pd.DataFrame (dataframe with test data)
    cat_feat : list (names of categorical features, default=[]).
    encoder : sklearn.preprocessing._encoders.OneHotEncoder (trained sklearn OneHotEncoder)
    lb : sklearn.preprocessing._label.LabelBinarizer (trained sklearn LabelBinarizer)
    label : str (name of label column "test")
    '''
    from data import process_data
    data_sorted = test.sort_values(by = [feature])
    names_feat, prec, rec, f1, acc = array([]), array([]), array([]), array([]), array([])
    l = 0 

    for i in range(1, len(data_sorted)):
        cond1 = (data_sorted[feature].values)[i-1]!= (data_sorted[feature].values)[i]
        cond2 = i== len(data_sorted)-1
        if cond1 or cond2:
            test_slice = test[l:i]
            X_test, y_test, _, _ = process_data(X=test_slice, categorical_features=cat_feat, 
                                                label=label, training=False, encoder=encoder, lb=lb)
            preds = inference(model, X_test)
            prec_, rec_, f1_, acc_ = compute_model_metrics(y_test, preds)
            prec = append(prec, prec_)
            rec = append(rec, rec_)
            f1 = append(f1, f1_)
            acc = append(acc, acc_)
            names_feat = append(names_feat, (data_sorted[feature].values)[i-1])
            l += 1

    metrics = array([prec, rec, f1, acc]).T
    columns = ["prec", "rec", "f1", "acc"]
    metrics_on_slice = DataFrame(data = metrics, columns=columns, index=names_feat)
    return metrics_on_slice