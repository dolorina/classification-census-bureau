from tkinter import Y
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from numpy import append, array, argmax, argmin, zeros, bincount
from data import process_data

# # Optional: implement hyperparameter tuning.
# def train_model(X_train, y_train):
#     """
#     Trains a machine learning model and returns it.
# 
#     Inputs
#     ------
#     X_train : np.array
#         Training data.
#     y_train : np.array
#         Labels.
#     Returns
#     -------
#     model
#         Trained machine learning model.
#     """
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)
#     
#     model_names = ["Random forest", "SVM", "Gaussian naive bayes", "Decision tree", "Neural network"]
#     models = [
#         RandomForestClassifier(max_depth=2, random_state=0), 
#         SVC(), 
#         GaussianNB(), 
#         DecisionTreeClassifier(max_depth=5), 
#         MLPClassifier(alpha=0.5, max_iter=500, learning_rate_init=0.01)]   
#     trained_models=[]
#     acc, prec, rec, f1 = zeros(len(models)), zeros(len(models)), zeros(len(models)), zeros(len(models))
#     
#     print()
#     print("Train and evaluate different classifiers.")
#     print()
#     i = 0
#     for cls in models:
#         cls.fit(X_train, y_train)
#         preds = inference(cls, X_val)
#         acc[i], prec[i], rec[i], f1[i] = compute_model_metrics(y_val, preds)
#         trained_models.append(cls)
#         print("    {}:".format(model_names[i]))
#         print("        acc={}, prec={}, rec={}, f1={}".format(round(acc[i],3), round(prec[i],3), round(rec[i],3), round(f1[i],3)))
#         i += 1
#     
#     maxima = array([argmax(acc), argmax(prec), argmax(rec), argmin(f1)])    
#     best_model = argmax(bincount(maxima))
#     if best_model.shape != 1: best_model = int(argmax(prec)) 
#     
#     print()
#     print("    Best performing model: {}.".format(model_names[best_model]))
#     print()
#     return trained_models[best_model]

# Optional: implement hyperparameter tuning.
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
    # parameter_space = {
    #     'hidden_layer_sizes': [(200, 100, 10,), (100, 50, 10,), (100,), (20,), (10,)],
    #     'activation': ['relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.001, 0.0001],
    #     'learning_rate': ['constant','adaptive'],
    # }
    parameter_space = {
        'hidden_layer_sizes': [(200, 100, 10,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.001],
        'learning_rate': ['constant'],
    }
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
    return model




def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    accuracy = accuracy_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    print(classification_report(y, preds))
    return accuracy, precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
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
