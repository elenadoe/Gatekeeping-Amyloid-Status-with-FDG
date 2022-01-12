#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:02:28 2021

@author: doeringe
"""

import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_classif


def extract_data(fdg_data, adni_data, maj_class, apoe_of_interest):
    """
    Extract data for specific group of interest (APOE4-nc or APOE4-c).

    First step of analyses.

    Parameters
    ----------
    fdg_data : csv table
        Input features: mean FDG-PET in 90 regions for all available subjects
        (outlier analysis and exclusion must be done prior to this step)
        Columns must be named PTID for id, APOE4 for apoe, AGE for age
        and PTGENDER for sex.
    adni_data : csv table
        Data table including apoe, age and sex information for all subjects.
    maj_class : int
        Whether majority class is 0 (AB-) or 1 (AB+)
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c are investigated.

    Returns
    -------
    subject_vectors : list
        Input features: mean FDG-PET in 90 regions for all included subjets
    y_true : list
        Amyloid status as assessed with gold-standard methods (ground truth)
    ids : list
        IDs of all participants
    gender : list
        Sex of all participants. 1 = Female, 0 = Male
    age : list
        Age of all participants.
    """
    names = fdg_data.columns.tolist()[1:]
    pickle.dump(names, open("../config/region_names.p", "wb"))
    labels = []
    y_true = []
    age = []
    apoe = []
    gender = []

    min_class = np.abs(maj_class-1)

    # GET ID, AGE, APOE CARRIERSHIP AND SEX
    for index, row in fdg_data.iterrows():
        pat = row[0]
        if adni_data['PTID'].str.contains(pat).any():
            # for statistics
            id_ = adni_data['PTID'] == pat
            age.append(adni_data['AGE'][id_].values[0])
            apoe.append(adni_data['APOE4'][id_].values[0])
            gender.append(adni_data['PTGENDER'][id_].values[0])

            # amyloid status
            val = adni_data['Amyloid_Status'][id_].values[0]
            labels.append(val)
            if val == 1:
                y_true.append(maj_class)
            elif val == 0:
                y_true.append(min_class)
    apoe = [int(x) for x in apoe]

    # CREATE SUBJECT MATRIX (samples x features)
    subject_vectors = []
    ids = []

    for index, row in fdg_data.iterrows():
        pat = row[0]

        if adni_data['PTID'].str.contains(pat).any():
            subject_vectors.append(row[1:].tolist())
            ids.append(row[0])

    # REDUCE FEATURE SET TO APOE CARRIERSHIP GROUP OF INTEREST
    if apoe_of_interest == 0:
        subject_vectors = [subject_vectors[x]
                           for x in range(len(subject_vectors))
                           if ((apoe[x] == 0) and (~np.isnan(age[x])))]
        y_true = [y_true[x] for x in range(len(y_true))
                  if ((apoe[x] == 0) and (~np.isnan(age[x])))]
        ids = [ids[x] for x in range(len(ids))
               if ((apoe[x] == 0) and (~np.isnan(age[x])))]
        gender = [gender[x] for x in range(len(gender))
                  if ((apoe[x] == 0) and (~np.isnan(age[x])))]
        gender = [1 if x == "Female" else 0 for x in gender]
        age = [age[x] for x in range(len(age))
               if ((apoe[x] == 0) and (~np.isnan(age[x])))]
    else:
        subject_vectors = [subject_vectors[x]
                           for x in range(len(subject_vectors))
                           if ((apoe[x] > 0) and (~np.isnan(age[x])))]
        y_true = [y_true[x] for x in range(len(y_true))
                  if ((apoe[x] > 0) and (~np.isnan(age[x])))]
        ids = [ids[x] for x in range(len(ids))
               if ((apoe[x] > 0) and (~np.isnan(age[x])))]
        gender = [gender[x] for x in range(len(gender))
                  if ((apoe[x] > 0) and (~np.isnan(age[x])))]
        gender = [1 if x == "Female" else 0 for x in gender]
        age = [age[x] for x in range(len(age))
               if ((apoe[x] > 0) and (~np.isnan(age[x])))]

    print("\033[1mParticipant Overview\033[0m\n",
          len(y_true), "subjects in this group.")
    print(np.where(np.array(y_true) == maj_class)[0].shape[0],
          " amyloid positive participants")
    print(np.where(np.array(y_true) == min_class)[0].shape[0],
          " amyloid negative participants")
    print("Mean age: {} years (SD = {}).".format(np.round(np.mean(age), 2),
                                                 np.round(np.std(age), 2)))

    return subject_vectors, y_true, ids, gender, age


def scale_data(data_train, data_test, var_names,
               apoe_of_interest, scaler=StandardScaler()):
    """
    Scale data.

    Scale data in train set and apply transformation parameters to test set.

    Parameters
    ----------
    data_train : list
        List of train input features to be scaled,
        e.g. [[90 FDG features], age]
    data_test : list
        List of test input features on which the inferred
        scaling parameters are to be applied.
    var_names : list
        List of variable names as strings.
    scaler : sklearn scaler object, optional
        How data is to be scaled. The default is StandardScaler.

    Returns
    -------
    data_train_scaled : list
        List of scaled train input features,
        e.g., [[90 scaled FDG features], scaled age]
    data_test_scaled : list
        List of scaled train input features

    """
    data_train_scaled = []
    data_test_scaled = []

    for i in range(len(data_train)):
        scaler = scaler
        d = np.array(data_train[i])
        d_test = np.array(data_test[i])
        if np.ndim(d) < 2:
            d = d.reshape(-1, 1)
            d_test = d_test.reshape(-1, 1)
        d_train = scaler.fit_transform(d)
        d_test = scaler.transform(d_test)
        pickle.dump(scaler, open("../config/{}_scaler_{}.p".format(
            var_names[i], apoe_of_interest), "wb"))
        data_train_scaled.append(d_train)
        data_test_scaled.append(d_test)

    return data_train_scaled, data_test_scaled


def feature_transform(X_train, y_train, X_test, apoe_of_interest,
                      fun_select=mutual_info_classif,
                      percentile=100, save=False):
    """
    Reduces feature set to best percentile according to fun_select.

    Parameters
    ----------
    X_train : list or np.array
        Input features: mean FDG-PET in 90 regions, age and sex
        for all subjects of the train set
    y_train : list or np.array
        Amyloid status as assessed with gold-standard methods (ground truth)
        in train set.
    X_test : list or np.array
        Input features: mean FDG-PET in 90 regions, age and sex
        for all subjets of the test set
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    fun_select : sklearn feature selection algorithm, optional
        Algorithm to use for evaluation of ad-hoc feature importance.
        The default is mutual_info_classif.
    percentile : int or float, optional
        Proportion of most important features to keep. The default is 100.
    save : boolean, optional
        Whether or not to save feature transformer

    Returns
    -------
    features_train : list or np.array
        Reduced input features for train set
    features_test : list or np.array
        Reduced input features for test set

    """
    feature_selector = SelectPercentile(
        score_func=fun_select, percentile=percentile)
    features_train = feature_selector.fit_transform(X_train, y_train)
    features_test = feature_selector.transform(X_test)

    if save:
        pickle.dump(feature_selector,
                    open("../config/feature_select_{}.p".format(apoe_of_interest),
                         "wb"))

    return features_train, features_test
