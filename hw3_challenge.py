# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn import metrics, exceptions
import hw3_main
from helper import *

def generate_feature_vector_challenge(df):
    return hw3_main.generate_feature_vector(df)

def impute_missing_values_challenge(X):
    return hw3_main.impute_missing_values(X)

def normalize_feature_matrix_challenge(X):
    return hw3_main.normalize_feature_matrix(X)

def select_C_challenge(X, y, C_range, penalty='l2', k=5, metric='accuracy'):
    return hw3_main.select_C(X, y, C_range, penalty, k, metric)


def run_challenge(X_challenge, y_challenge, X_heldout):
    # TODO:
    # Read challenge data
    # Train a linear classifier and apply to heldout dataset features
    # Use generate_challenge_labels to print the predicted labels
    print("================= Part 3 ===================")
    print("Part 3: Challenge")
    #C_range = np.logspace(-3, 3, 7)
    #metric_list = ["f1_score", "auroc"]
    #for metric in metric_list:
    #    best_C = select_C_challenge(X_challenge, y_challenge, C_range, 'l2', 5, metric)
    #    print("metric: " + metric + " Best C: %.6f" % best_C)
    clf = LogisticRegression() 
    clf = get_classifier(penalty='l1', C=1000)
    best_C = 100
    clf = get_classifier(penalty='l2', C=best_C, class_weight='balanced')
    clf.fit(X_challenge, y_challenge)

    #confusion matrix on the training data X_challenge, y_challenge
    cm = metrics.confusion_matrix(y_challenge, clf.predict(X_challenge), labels=[1, -1])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print("True Negatives: %d" % tn)
    print("False Positives: %d" % fp)
    print("False Negatives: %d" % fn)
    print("True Positives: %d" % tp)

    # predict the labels for the heldout data
    y_score = clf.predict_proba(X_heldout)[:, 1]
    y_label = clf.predict(X_heldout)
    make_challenge_submission(y_label, y_score)


if __name__ == '__main__':
    # Read challenge data
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    # TODO: Question 3: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    run_challenge(X_challenge, y_challenge, X_heldout)
    test_challenge_output()
