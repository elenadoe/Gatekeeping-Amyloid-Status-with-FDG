#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:39:33 2022

@author: doeringe
"""
import transform_data
import show_performance
import pickle
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import fbeta_score


def test_external(df_fdg, df_data, apoe_of_interest, scoring='f_tenth',
                  info=True):
    """
    Test trained classifier on external dataset.

    Transforms features conforming to standard established during
    training on ADNI data and subsequently yields predictions from
    the final model on the external test set.

    Parameters
    ----------
    df_fdg : csv table
        Input features: mean FDG-PET in 90 regions for all available subjects
        (outlier analysis and exclusion must be done prior to this step)
    df_data :  csv table
        Data table including id, apoe, age and sex information
        for all subjects.
        Columns must be named PTID for id, APOE4 for apoe, AGE for age
        and PTGENDER for sex.
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    scoring : str or f_tenth
        Should be one out of: "f1", "accuracy", "balanced_accuracy" or f_tenth.
        The default is 'f_tenth'.
    info : boolean, optional
        Whether to provide information on internal stats.
        The default is True.

    Returns
    -------
    None.

    """
    if apoe_of_interest == 0:
        maj_class = 0
    elif apoe_of_interest > 0:
        maj_class = 1

    subject_vectors, y_true, ids, gender, age = transform_data.extract_data(
        df_fdg, df_data, maj_class=maj_class,
        apoe_of_interest=apoe_of_interest)

    if info:
        print("EXTERNAL DATASET:\nn_minority class: ",
              np.bincount(y_true)[0],
              "n_majority class: ", np.bincount(y_true)[1])

    # FEATURE SCALING
    feature_scaler = pickle.load(open(
        "../config/X_train_scaler_{}.p".format(apoe_of_interest), "rb"))
    age_scaler = pickle.load(open(
        "../config/age_train_scaler_{}.p".format(apoe_of_interest), "rb"))
    X = feature_scaler.transform(subject_vectors)
    age = age_scaler.transform(np.array(age).reshape(-1, 1))

    X = np.concatenate((X, age, np.array(gender).reshape(-1, 1)), axis=1)

    # FEATURE SELECTION
    feature_select = pickle.load(
        open("../config/feature_select_{}.p".format(apoe_of_interest), "rb"))
    X = feature_select.transform(X)
    pickle.dump(X, open("../results/features/IMC_features_{}.p".format(
        apoe_of_interest), "wb"))
    pickle.dump(y_true, open("../results/features/IMC_labels_{}.p".format(
        apoe_of_interest), "wb"))

    # PREDICTION
    model = pickle.load(open(
        "../results/final_model_{}.p".format(apoe_of_interest), "rb"))
    pred = model.predict(X)

    if info:
        show_performance.show_performance(y_true, pred)


def test_balanced(dataset, apoe_of_interest, scoring='f_tenth'):
    """
    Test trained classifier on randomly downsampled, balanced test sets.

    Parameters
    ----------
    dataset : str
        Which dataset to assess. Readily transformed features and labels
        have already been saved during previous analyses and are called
        from disk.
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    scoring : str or f_tenth
        Should be one out of: "f1", "accuracy", "balanced_accuracy" or f_tenth.
        The default is 'f_tenth'.

    Returns
    -------
    None.

    """
    X = pickle.load(
        open("../results/features/{}_features_{}.p".format(
            dataset, apoe_of_interest), "rb"))
    y_true = pickle.load(
        open("../results/features/{}_labels_{}.p".format(
            dataset, apoe_of_interest), "rb"))

    # PREDICTION ON RANDOMLY GENERATED BALANCED SUBSETS
    model = pickle.load(open(
        "../results/final_model_{}.p".format(apoe_of_interest), "rb"))
    maj_ = np.where(np.array(y_true) == 1)
    min_ = np.where(np.array(y_true) == 0)

    f_ten_balanced = []

    for i in range((len(min_[0]*2)**2)):
        maj_test_res, y_maj_res = resample(
            np.array(X)[maj_], np.array(y_true)[maj_],
            n_samples=len(min_[0]), replace=False, random_state=i)
        features_test_balanced = np.concatenate(
            (np.array(X)[min_], maj_test_res))
        y_true_balanced = np.concatenate(
            (np.array(y_true)[min_], y_maj_res))
        pred = model.predict(features_test_balanced)
        f_ten_balanced.append(
            fbeta_score(y_true_balanced, pred, beta=1/10))
    print("Mean: {} (range: {} - {})".format(
        np.round(np.mean(f_ten_balanced), 3),
        np.round(np.min(f_ten_balanced), 3),
        np.round(np.max(f_ten_balanced), 3)))
