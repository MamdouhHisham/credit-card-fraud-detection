import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer , f1_score
from scipy.stats import uniform
from scripts.Model_Evaluation import *
import pickle


def logistic_regression_tuning(X_train, y_train, X_val, y_val):
    param = {
        'penalty': ['l1','l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'newton-cholesky', 'lbfgs', 'saga', 'liblinear', 'sag'],
        'max_iter': [500, 800,1000, 1500, 2000],
        'C': uniform(1,10),
        'class_weight': ['balanced',
                         {0: 0.35, 1: 0.65},
                         {0: 0.25, 1: 0.75},
                         {0: 0.15, 1: 0.85},
                         {0: 0.10, 1: 0.90}]
    }

    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    model = LogisticRegression()

    random_search = RandomizedSearchCV(model, param_distributions= param,
                                       cv=kfold, n_jobs=-1, verbose=2)

    random_search.fit(X_train, y_train)
    print("Best Parameters: ", random_search.best_params_)

    evaluate_model(random_search, X_train, y_train, X_val, y_val, threshold=0.5)

    evaluate_model(random_search, X_train, y_train, X_val, y_val, threshold='best')

    return random_search.best_params_

def logistic_regression_training(X_train, y_train, X_val, y_val, best_threshold = True):

    LR = LogisticRegression(max_iter= 800,
                            solver='newton-cg',
                            penalty='l2',
                            class_weight= {0: 0.35, 1: 0.65},
                            C = 6.414251027864595,
                            random_state=42)

    LR.fit(X_train, y_train)

    if best_threshold:
        evaluate_model(LR, X_train, y_train, X_val, y_val, threshold='best',  model_name="Logistic Regression")
    else:
        evaluate_model(LR, X_train, y_train, X_val, y_val, threshold='0.5')

    with open('../models/LogisticRegression_model.pkl', 'wb') as model_file:
        pickle.dump(LR, model_file)
    print("Model saved as 'LogisticRegression_model'.")

    return LR


def RandomForest_tuning(X_train, y_train, X_val, y_val):

    param = {
        'n_estimators': [100, 200, 300,400],
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'class_weight': ['balanced_subsample', {0: 0.35, 1: 0.65}, {0: 0.20, 1: 0.80}]
    }

    kfold = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    rf = RandomForestClassifier()

    random_search = RandomizedSearchCV(rf,
                                     param_distributions= param,
                                     cv=kfold,
                                     n_jobs=-1,
                                     n_iter=30,
                                     verbose=2,
                                     random_state=42)

    random_search.fit(X_train, y_train)
    print("Best Parameters: ", random_search.best_params_)

    eval = evaluate_model(random_search, X_train, y_train, X_val, y_val, threshold=0.5)

    evaluate_model(random_search, X_train, y_train, X_val, y_val, threshold='best')

    return eval


def RandomForest_training(X_train, y_train, X_val, y_val, best_threshold = True):

    RF = RandomForestClassifier(n_estimators=300,
                                min_samples_split=10,
                                min_samples_leaf=10,
                                max_depth=15,
                                class_weight={0: 0.2, 1: 0.8},
                                random_state=42,
                                n_jobs=-1)

    RF.fit(X_train, y_train)

    if best_threshold:
        evaluate_model(RF, X_train, y_train, X_val, y_val, threshold='best',model_name="Random Forest")
    else:
        evaluate_model(RF, X_train, y_train, X_val, y_val, threshold= 0.5)

    with open('../models/RandomForest_model.pkl', 'wb') as model_file:
        pickle.dump(RF, model_file)
    print("Model saved as 'RandomForest_model'.")

    return RF

def NeuralNet_tuning(X_train, y_train, X_val, y_val):

    param = {
        'activation': ['relu', 'tanh'],
        'hidden_layer_sizes': [
            (20, 10),
            (30, 20),
            (30, 20, 10),
            (40, 30, 20),
            (64, 32, 32, 16)
    ],
        'solver': ['adam', 'sgd'],
        'batch_size': [32, 64, 128, 512],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'alpha': [0.001, 0.01, 0.1],
        'max_iter': [500, 800, 1000, 2000],
        'random_state': [42],
        'early_stopping': [True],
        'validation_fraction': [0.1],
    }

    kfold = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    mlp = MLPClassifier()
    random_search = RandomizedSearchCV(mlp,
                                       param_distributions=param,
                                       n_jobs=-1,
                                       n_iter=30,
                                       random_state=42,
                                       cv=kfold,
                                       verbose=2,
                                       )
    random_search.fit(X_train, y_train)
    print("Best Parameters: ", random_search.best_params_)

    eval = evaluate_model(random_search, X_train, y_train, X_val, y_val, threshold=0.5)

    return eval

def NeuralNet_training(X_train, y_train, X_val, y_val, best_threshold = True):

    mlp = MLPClassifier(validation_fraction=0.1,
                        solver='sgd',
                        random_state=42,
                        max_iter=800,
                        learning_rate_init=0.01,
                        hidden_layer_sizes=(40,30,20),
                        early_stopping=True,
                        batch_size=128,
                        alpha=0.001,
                        activation='relu')

    mlp.fit(X_train, y_train)

    if best_threshold:
        evaluate_model(mlp, X_train, y_train, X_val, y_val, threshold='best', model_name="Neural Network")
    else:
        evaluate_model(mlp, X_train, y_train, X_val, y_val, threshold='0.5')

    with open('../models/NeuralNetwork_model.pkl', 'wb') as model_file:
        pickle.dump(mlp, model_file)
    print("Model saved as 'NeuralNetwork_model'.")

    return mlp