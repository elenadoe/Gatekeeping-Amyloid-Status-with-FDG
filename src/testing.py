#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:39:33 2022

@author: doeringe
"""
import transform_data
import show_performance
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, precision_score, recall_score,\
    confusion_matrix
from imblearn.under_sampling import RandomUnderSampler


def test_external(df_fdg, df_data, apoe_of_interest, percentile, try_,
                  drop_features=None, scoring='f_tenth', info=True):
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
        pos = 0
    elif apoe_of_interest > 0:
        pos = 1

    # df_data = df_data[df_data['IN_RANGE'] == 1]
    # df_data.reset_index()
    subject_vectors, y_true, ids, gender, age,\
        apoe, av45, mmse, race, cdr = transform_data.extract_data(
            df_fdg, df_data, pos=pos,
            apoe_of_interest=apoe_of_interest, what="IMC")

    if info:
        print("EXTERNAL DATASET:\nn_minority class: ",
              np.bincount(y_true)[0],
              "n_majority class: ", np.bincount(y_true)[1])
        transform_data.info_on(y_true, [], age, [],
                               gender, [], gender, [], gender, [],
                               cdr, [], what="IMC")

    # FEATURE SCALING
    feature_scaler = pickle.load(
        open("../config/X_train_scaler_{}_{}{}.p".format(
            percentile, apoe_of_interest, try_), "rb"))
    age_scaler = pickle.load(
        open("../config/age_train_scaler_{}_{}{}.p".format(
            percentile, apoe_of_interest, try_), "rb"))
    X = feature_scaler.transform(subject_vectors)
    age = age_scaler.transform(np.array(age).reshape(-1, 1))

    X = np.concatenate((X, age, np.array(gender).reshape(-1, 1)), axis=1)
    print("Shape of X: ", X.shape)

    if drop_features == "fdg":
        X = X[:, [-2, -1]]
    elif drop_features == "age":
        X = X[:, list(range(90)) + [-1]]
    elif drop_features == "gender":
        X = X[:, list(range(90)) + [-2]]
    print("Drop features: {}".format(drop_features),
          "Shape of X_train: ", X.shape)

    # FEATURE SELECTION
    feature_select = pickle.load(
        open("../config/feature_select_{}_{}_wo-{}{}.p".format(
            percentile, apoe_of_interest, drop_features, try_), "rb"))
    X = feature_select.transform(X)
    pickle.dump(X,
                open("../results/features/IMC_features_{}_{}_wo-{}{}.p".format(
                    percentile, apoe_of_interest, drop_features, try_), "wb"))
    pickle.dump(y_true,
                open("../results/features/IMC_labels_{}_{}{}.p".format(
                    percentile, apoe_of_interest, try_), "wb"))

    # PREDICTION
    model = pickle.load(open(
        "../results/final_model_{}_{}_wo-{}{}.p".format(percentile,
                                                      apoe_of_interest,
                                                      drop_features,
                                                      try_), "rb"))
    print(model)
    pred = model.predict(X)

    if info:
        show_performance.show_performance(y_true, pred)


def show_mlp_loss(apoe_of_interest, percentile, try_, drop_features=None):
    model = pickle.load(open(
        "../results/final_model_{}_{}_wo-{}{}.p".format(percentile,
                                                        apoe_of_interest,
                                                        drop_features,
                                                        try_), "rb"))
    plt.plot(model.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def test_balanced(dataset, apoe_of_interest, percentile, drop_features, try_,
                  random_state, scoring='f_tenth'):
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
        open("../results/features/{}_features_{}_{}_wo-{}{}.p".format(
            dataset, percentile, apoe_of_interest, drop_features, try_), "rb"))
    y_true = pickle.load(
        open("../results/features/{}_labels_{}_{}{}.p".format(
            dataset, percentile, apoe_of_interest, try_), "rb"))

    # PREDICTION ON RANDOMLY GENERATED BALANCED SUBSETS
    model = pickle.load(open(
        "../results/final_model_{}_{}_wo-{}{}.p".format(percentile,
                                                        apoe_of_interest,
                                                        drop_features,
                                                        try_), "rb"))

    n_min = np.bincount(y_true)[0]
    fbeta = []
    prec = []
    rec = []
    for n in range(n_min**2):
        rus = RandomUnderSampler(random_state=n)
        X_balanced, y_balanced = rus.fit_resample(X, y_true)
    
        pred = model.predict(X_balanced)
        fbeta.append(fbeta_score(y_balanced, pred, beta=1/10))
        prec.append(precision_score(y_balanced, pred))
        rec.append(recall_score(y_balanced, pred))
    print(str(n_min**2), " randomly downsampled samples drawn",
          "from majority class to n =", np.bincount(y_balanced)[0])

    print("F1/10: {} (Precision: {}, Recall: {})".format(
        np.round(np.mean(fbeta), 3),
        np.round(np.mean(prec), 3),
        np.round(np.mean(rec), 3)))

