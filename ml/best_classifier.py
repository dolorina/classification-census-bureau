'''
Script that contains a function that tests different classifier models
The best performing model was the MLP classifier. 

Author: Marina Dolokov
Date: Februar 2022
'''

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from model import inference, compute_model_metrics
from data import process_data
from numpy import append, array, argmax, argmin, zeros, bincount


def train_model(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)
    
    model_names = ["Random forest", "SVM", "Gaussian naive bayes", "Decision tree", "Neural network"]
    models = [
        RandomForestClassifier(max_depth=2, random_state=0), 
        SVC(), 
        GaussianNB(), 
        DecisionTreeClassifier(max_depth=5), 
        MLPClassifier(alpha=0.5, max_iter=500, learning_rate_init=0.01)]   
    trained_models=[]
    acc, prec, rec, f1 = zeros(len(models)), zeros(len(models)), zeros(len(models)), zeros(len(models))
    
    print()
    print("Train and evaluate different classifiers.")
    print()
    i = 0
    for cls in models:
        cls.fit(X_train, y_train)
        preds = inference(cls, X_val)
        acc[i], prec[i], rec[i], f1[i] = compute_model_metrics(y_val, preds)
        trained_models.append(cls)
        print("    {}:".format(model_names[i]))
        print("        acc={}, prec={}, rec={}, f1={}".format(round(acc[i],3), round(prec[i],3), round(rec[i],3), round(f1[i],3)))
        i += 1
    
    maxima = array([argmax(acc), argmax(prec), argmax(rec), argmin(f1)])    
    best_model = argmax(bincount(maxima))
    if best_model.shape != 1: best_model = int(argmax(prec)) 
    
    print()
    print("    Best performing model: {}.".format(model_names[best_model]))
    print()
    return trained_models[best_model]
