#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:02:28 2021

@author: doeringe
"""

import numpy as np
import pandas as pd
import pickle
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_classif


def extract_data(fdg_data, adni_data, pos, apoe_of_interest, what="ADNI",
                 info=True):
    """
    Extract data for specific group of interest (APOE4-nc or APOE4-c).

    First step of analyses.

    Parameters
    ----------
    fdg_data : pd.DataFrame
        Input features: mean FDG-PET in 90 regions for all available subjects
        (outlier analysis and exclusion must be done prior to this step)
        Columns must be named PTID for id, APOE4 for apoe, AGE for age
        and PTGENDER for sex.
    adni_data : csv table
        Data table including apoe, age and sex information for all subjects.
    drop_features : str
        Whether and which features to drop (for ablation study)
    pos : int
        Whether amyloid positivity should be classified (1)
        or not (amyloid negativity of interest; 0)
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) or all (-1) are investigated.

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
    race = []
    av45 = []
    mmse = []
    cdr = []

    neg = np.abs(pos-1)

    # GET ID, AGE, APOE CARRIERSHIP AND SEX
    for index, row in fdg_data.iterrows():
        pat = row[0]
        if adni_data['PTID'].str.contains(pat).any():
            # for statistics
            id_ = adni_data['PTID'] == pat
            age.append(adni_data['AGE'][id_].values[0])
            apoe.append(adni_data['APOE4'][id_].values[0])
            gender.append(adni_data['PTGENDER'][id_].values[0])
            cdr.append(adni_data['CDRSB'][id_].values[0])
            if what == "ADNI":
                mmse.append(adni_data['MMSE'][id_].values[0])
                race.append(adni_data['PTRACCAT'][id_].values[0])
                av45.append(adni_data['AV45'][id_].values[0])

            # amyloid status
            # precision of "1" values is evaluated
            # for APOE4-nc: append 1 if amyloid status == 0, else append 0
            # for APOE4-c: append 1 if amyloid status == 1, else append 0
            val = adni_data['AMY_STAT'][id_].values[0]
            labels.append(val)
            if val == 1:
                y_true.append(pos)
            elif val == 0:
                y_true.append(neg)
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
        if what == "ADNI":
            race = [race[x] for x in range(len(race))
                    if ((apoe[x] == 0) and (~np.isnan(age[x])))]
            av45 = [av45[x] for x in range(len(av45))
                    if ((apoe[x] == 0) and (~np.isnan(age[x])))]
            mmse = [mmse[x] for x in range(len(mmse))
                    if ((apoe[x] == 0) and (~np.isnan(age[x])))]
        cdr = [cdr[x] for x in range(len(cdr))
               if ((apoe[x] == 0) and (~np.isnan(age[x])))]
        age = [age[x] for x in range(len(age))
               if ((apoe[x] == 0) and (~np.isnan(age[x])))]
    elif apoe_of_interest == 1:
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
        if what == "ADNI":
            race = [race[x] for x in range(len(race))
                    if ((apoe[x] > 0) and (~np.isnan(age[x])))]
            av45 = [av45[x] for x in range(len(av45))
                    if ((apoe[x] > 0) and (~np.isnan(age[x])))]
            mmse = [mmse[x] for x in range(len(mmse))
                    if ((apoe[x] > 0) and (~np.isnan(age[x])))]
        cdr = [cdr[x] for x in range(len(cdr))
               if ((apoe[x] > 0) and (~np.isnan(age[x])))]
        age = [age[x] for x in range(len(age))
               if ((apoe[x] > 0) and (~np.isnan(age[x])))]
    elif apoe_of_interest == -1:
        subject_vectors = [subject_vectors[x]
                           for x in range(len(subject_vectors))
                           if (~np.isnan(age[x]))]
        y_true = [y_true[x] for x in range(len(y_true))
                  if (~np.isnan(age[x]))]
        ids = [ids[x] for x in range(len(ids))
               if (~np.isnan(age[x]))]
        gender = [gender[x] for x in range(len(gender))
                  if (~np.isnan(age[x]))]
        gender = [1 if x == "Female" else 0 for x in gender]
        if what == "ADNI":
            race = [race[x] for x in range(len(race))
                    if (~np.isnan(age[x]))]
            av45 = [av45[x] for x in range(len(av45))
                    if (~np.isnan(age[x]))]
            mmse = [mmse[x] for x in range(len(mmse))
                    if (~np.isnan(age[x]))]
        cdr = [cdr[x] for x in range(len(cdr))
               if (~np.isnan(age[x]))]
        apoe = [apoe[x] for x in range(len(apoe))
                if (~np.isnan(age[x]))]
        age = [age[x] for x in range(len(age))
               if (~np.isnan(age[x]))]
    if info:
        print("\033[1mParticipant Overview\033[0m\n",
              len(y_true), "subjects in this group.")
        print(np.where(np.array(y_true) == pos)[0].shape[0],
              " amyloid positive participants")
        print(np.where(np.array(y_true) == neg)[0].shape[0],
              " amyloid negative participants")
        print("Mean age: {} years (SD = {}, range: {} - {} years).".format(
            np.round(np.mean(age), 2), np.round(np.std(age), 2),
            np.min(age), np.max(age)))
        print("CDR_SOB", np.mean(cdr), np.std(cdr))

    return subject_vectors, y_true, ids, gender, age, apoe,\
        av45, mmse, race, cdr

def info_on(y_train, y_test, age_train, age_test, gender_train, gender_test,
            race_train, race_test, av45_train, av45_test,
            cdr_train, cdr_test, what="ADNI"):
    train = {'labels': [str(y) for y in y_train], 'age': age_train,
             'sex': [str(g) for g in gender_train],
             'race': race_train, 'av45': av45_train, 'cdr': cdr_train}
    test = {'labels': [str(y) for y in y_test], 'age': age_test,
            'sex': [str(g) for g in gender_test],
            'race': race_test, 'av45': av45_test, 'cdr': cdr_test}

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    if what == "ADNI":
        print("\033[1mTRAIN STATISTICS\nMajority Class\033[0m\n",
              train[train['labels'] == "1"].describe(include="all"),
              "\n\033[1mMinority Class\033[0m\n",
              train[train['labels'] == "0"].describe(include="all"))
        print("\033[1mTEST STATISTICS\nMajority Class\033[0m\n",
              test[test['labels'] == "1"].describe(include="all"),
              "\n\033[1mMinority Class\033[0m\n",
              test[test['labels'] == "0"].describe(include="all"))
    else:
        print("\033[1mPARTICIPANT STATISTICS\nMajority Class\033[0m\n",
              train[train['labels'] == "1"].describe(include="all"),
              "\n\033[1mMinority Class\033[0m\n",
              train[train['labels'] == "0"].describe(include="all"))


def scale_data(data_train, data_test, var_names,
               apoe_of_interest, percentile, try_, scaler=StandardScaler()):
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
        pickle.dump(scaler, open("../config/{}_scaler_{}_{}{}.p".format(
            var_names[i], percentile, apoe_of_interest, try_), "wb"))
        data_train_scaled.append(d_train)
        data_test_scaled.append(d_test)

    return data_train_scaled, data_test_scaled


def feature_transform(X_train, y_train, X_test, apoe_of_interest, percentile,
                      try_, fun_select=mutual_info_classif, 
                      drop_features=None, save=False):
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
                    open("../config/feature_select_{}_{}_wo-{}{}.p".format(
                        percentile, apoe_of_interest,
                        drop_features, try_), "wb"))

    return features_train, features_test


def save_results(df_data, ids_test, pred, age_test, gender_test,
                 apoe_of_interest, try_, percentile=100,
                 drop_features=None):
    """
    Save results.

    Save results in csv file.

    Parameters
    ----------
    df_data : pd.DataFrame
        Dataframe containing ADNImerge info
    ids_test : list
        Test ids
    pred : list
        Predictions of test set
    age_test : list
        Age of individuals in test set
    gender_test : list
        Sex of individuals in test set
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c are investigated.

    Returns
    -------
    None.

    """
    results = {'PTID': ids_test, 'Age': age_test, 'Sex': gender_test,
               'pred': pred}
    results = pd.DataFrame(results)
    results_merged = results.merge(df_data, how="left", on="PTID")
    results_merged.to_csv(
        "../results/Predictions_ADNImerge_{}_{}_wo-{}{}.csv".format(
            percentile, apoe_of_interest, drop_features, try_))
