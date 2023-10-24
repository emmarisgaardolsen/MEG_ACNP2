#%% IMPORTS
import os
import mne
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.inspection import permutation_importance



#%% SIMPLE CLASSIFICATION

def get_indices(y, triggers):
    indices = list()
    for trigger_index, trigger in enumerate(y):
        if trigger in triggers:
            indices.append(trigger_index)
            
    return indices


def equalize_number_of_indices(X, y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()

    keep_inds = np.concatenate([np.random.choice(np.where(y == cls)[0], min_class_count, replace=False) for cls in unique_classes])

    X_equal = X[keep_inds, :, :]
    y_equal = y[keep_inds]

    return X_equal, y_equal


def simple_classification(X, y, triggers, penalty='None', C=1.0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import permutation_test_score

    n_samples = X.shape[2]
    indices = get_indices(y, triggers)

    X = X[indices, :, :]
    y = y[indices]

    X, y = equalize_number_of_indices(X, y)

    gnb = GaussianNB()
    sc = StandardScaler()
    cv = StratifiedKFold()

    mean_scores = np.zeros(n_samples)
    y_pred_all = []
    y_true_all = []
    permutation_scores = np.zeros((n_samples, 100))  # Adjust the number of permutations as needed
    pvalues = np.zeros(n_samples)
    feature_importance = np.zeros((X.shape[1], n_samples))

    for sample_index in range(n_samples):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)

        y_pred = cross_val_predict(gnb, this_X_std, y, cv=cv)
        y_true = y

        scores = np.mean(y_pred == y_true)
        mean_scores[sample_index] = scores

        y_pred_all.append(y_pred)
        y_true_all.append(y_true)

        # Permutation test
        _, permutation_score, pvalue = permutation_test_score(gnb, this_X_std, y, cv=cv)
        permutation_scores[sample_index, :] = permutation_score
        pvalues[sample_index] = pvalue

        # Feature importance using permutation importance
        gnb.fit(this_X_std, y)
        importances = permutation_importance(gnb, this_X_std, y)
        feature_importance[:, sample_index] = importances.importances_mean

        print(sample_index)

    return mean_scores, y_pred_all, y_true_all, permutation_scores, pvalues, feature_importance



