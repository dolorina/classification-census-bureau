from sklearn.metrics import make_scorer, accuracy_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

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
    
    # model = SVC(decision_function_shape='ovo') # GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score)
    # model.fit(X_train, y_train)

    # Set the parameters by cross-validation
    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]},
        {"kernel": ["linear"], "C": [1, 10]}, #, 100, 1000]},
    ]
    scores = {"accuracy": make_scorer(accuracy_score), "precision":  make_scorer(precision_score), "recall": make_scorer(recall_score)}

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        model = GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score)
        model.fit(X_train, y_train)
        
        print("Best parameters set found on development set:")
        print(model.best_params_)
        print()

        print("Grid scores on development set:")
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

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
