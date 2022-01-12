#!/usr/bin/env python
# coding: utf-8

import pickle
import transform_data
import show_performance
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score, \
    f1_score, fbeta_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle, resample
from tqdm import tqdm


# %%

f_tenth = make_scorer(fbeta_score, beta=0.1)


def evaluate(scoring, y_true, y_pred):
    """
    Yield scorer for test set.

    For simplification of terminology.

    Parameters
    ----------
    scoring : str or f_tenth
        Should be one out of: "f1", "accuracy", "balanced_accuracy" or f_tenth
    y_true : list or np.array
        Amyloid status as assessed with gold-standard methods (ground truth)
    y_pred : list or np.array
        Predicted amyloid status

    Raises
    ------
    Exception
        If none of the available scoring methods are chosen.

    Returns
    -------
    Score : float
        Performance score depending on chosen metric

    """
    if scoring == "f1":
        return(f1_score(y_true, y_pred))
    elif scoring == "accuracy":
        return(accuracy_score(y_true, y_pred))
    elif scoring == "balanced_accuracy":
        return(balanced_accuracy_score(y_true, y_pred))
    elif scoring == f_tenth:
        return(fbeta_score(y_true, y_pred, beta=0.1))
    else:
        raise Exception(scoring,
                        "not in list of score metrics" +
                        " acknowledged by program.")


def upsample(X_train, y_train, maj_class, info=True,
             random_state=42):
    """
    Automatic upsampling of minority class.

    Randomly generates copies of available data in minority class
    until sample size of minority and majority class are equal.
    This helps to avoid overfitting. Only works with binary classification.

    Parameters
    ----------
    X_train : list or np.array
        Input features: mean FDG-PET in 90 regions for all subjets
        of the train set
    y_train : list or np.array
        Amyloid status as assessed with gold-standard methods (ground truth)
        in train set.
    maj_class : int
        Whether majority class is 0 (AB-) or 1 (AB+)
    info : boolean, optional
        Whether to provide information on internal stats. The default is True.
    random_state : int, optional
        random seed. The default is 42.

    Returns
    -------
    X_train_res : list or np.array
        Upsampled input features
    y_train_res : list or np.array
        Upsampled ground truth

    """
    min_class = np.abs(maj_class-1)
    min_subjects = np.array(X_train)[np.where(np.array(y_train) == min_class)]
    min_y = [min_class]*min_subjects.shape[0]
    maj_subjects = np.array(X_train)[np.where(np.array(y_train) == maj_class)]
    maj_y = [maj_class]*maj_subjects.shape[0]

    # UPSAMPLE DEPENDING ON WHICH SAMPLE IS BIGGER
    if min_subjects.shape[0] > maj_subjects.shape[0]:
        maj_subjects_resampled, maj_y_resampled = resample(
            maj_subjects, maj_y, replace=True,
            n_samples=min_subjects.shape[0], random_state=random_state)
        X_train_res = np.concatenate((maj_subjects_resampled, min_subjects))
        y_train_res = np.concatenate((maj_y_resampled, min_y))
        X_train_res, y_train_res = shuffle(X_train_res, y_train_res,
                                           random_state=random_state)

    elif min_subjects.shape[0] < maj_subjects.shape[0]:
        min_subjects_resampled, min_y_resampled = resample(
            min_subjects, min_y, replace=True,
            n_samples=maj_subjects.shape[0], random_state=random_state)
        X_train_res = np.concatenate((maj_subjects, min_subjects_resampled))
        y_train_res = np.concatenate((maj_y, min_y_resampled))
        X_train_res, y_train_res = shuffle(X_train_res, y_train_res,
                                           random_state=random_state)

    elif min_subjects.shape[0] == maj_subjects.shape[0]:
        print("Sample size already equal")
        X_train_res = X_train
        y_train_res = y_train

    if info:
        print("\n", X_train_res.shape[0], "train samples after upsampling")
        print(np.bincount(y_train_res))

    return X_train_res, y_train_res


def classifier_feature_search_cv(X_train, y_train,
                                 X_test, y_test,
                                 model_list, model_names,
                                 params,
                                 scoring,
                                 apoe_of_interest,
                                 upsampling=True,
                                 percentiles=[100],
                                 cv=10,
                                 maj_class=1,
                                 random_state=42):
    """
    Hyperparameter search.

    Yields intermediate models and their
    performance during cross-validation on the training data,
    as well as prediction performance on test data

    Parameters
    ----------
    X_train : list or np.array
        Input features: mean FDG-PET in 90 regions, age and sex
        for all subjets of the train set
    y_train : list or np.array
        Amyloid status as assessed with gold-standard methods (ground truth)
        in train set.
    X_test : list or np.array
        Input features: mean FDG-PET in 90 regions for all subjets
        of the test set
    y_test : list or np.array
        Amyloid status as assessed with gold-standard methods (ground truth)
        in test set
    model_list : list
        contains sklearn algorihtms
    model_names : list
        contains names of models
    params : dict
        dictionary of hyperparameters to assess per classifier
    scoring : str or sklearn scoring metric
        Can be any sklearn scoring metric, or "f_tenth" for F_1/10 score
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    upsampling : boolean, optional
        Whether or not to use upsampling to avoid overfitting.
        The default is True.
    percentiles : int or float, optional
        Proportion of most important features to keep. The default is [100].
    cv : int, optional
        How many folds to split training data into for cross-validation.
        The default is 10.
    maj_class : int
        Whether majority class is 0 (AB-) or 1 (AB+)
    random_state : int, optional
        Random seed. The default is 42.

    Returns
    -------
    validation : 2-dimensional np.array
        Array containing scores of intermediate models
        obtained from cross-validation.
        Shape is percentilesxmodels.
    test : 2-dimensional np.array
        Array containing test scores of intermediate models
        obtained after cross-validation.
        Shape is percentilesxmodels.
    fitted_models : dict
        Dictionary containing all fitted intermediate models

    """
    if scoring == "f_tenth":
        scoring1 = f_tenth
    else:
        scoring1 = scoring

    # PREPARE DATA STORAGE
    fitted_models = {}
    validation = np.empty(shape=(len(model_list), len(percentiles)))
    test = np.empty(shape=(len(model_list), len(percentiles)))

    # ITERATE OVER ALL PERCENTILES OF INPUT FEATURES TO BE ASSESSED
    for p in percentiles:
        fitted_models_pp = []
        print("PERCENTILE: Upper {}% of training data".format(p))
        features_train, features_test = transform_data.feature_transform(
            X_train, y_train, X_test, apoe_of_interest)

        if upsampling:
            features_train, y_train_ups = upsample(features_train, y_train,
                                                   maj_class=maj_class)

        else:
            y_train_ups = y_train

        for i in tqdm(range(len(model_list))):
            classifier = model_list[i]
            clf = GridSearchCV(classifier, params[model_names[i]],
                               n_jobs=-1, cv=StratifiedKFold(n_splits=cv),
                               scoring=scoring1)
            clf.fit(features_train, y_train_ups)

            # VALIDATION PERFORMANCE
            validation[i, percentiles.index(p)] = clf.best_score_
            fitted_models_pp.append(clf.best_estimator_)

            # TEST PERFORMANCE
            pred = clf.predict(features_test)
            test[i, percentiles.index(p)] = evaluate(
                scoring1, y_test, pred)

        # SAVE TRAINED MODEL
        fitted_models[p] = fitted_models_pp

    return validation, test, fitted_models


def choose_best_model(validation, test, fitted_models, X_train,
                      y_train, X_test, apoe_of_interest, percentile=100):
    """
    Choose final model among intermediate models.

    Best model per percentile and final feature set is returned.

    Parameters
    ----------
    validation : 2-dimensional np.array
        Array containing scores of intermediate models
        obtained from cross-validation.
        Shape is percentilesxmodels, validation array yielded
        from classifier_feature_search.
    test : 2-dimensional np.array
        Array containing test scores of intermediate models
        obtained after cross-validation.
        Shape is percentilesxmodels, test array yielded
        from classifier_feature_search.
    fitted_models : dict
        Dictionary containing all fitted intermediate models
    X_train : list or np.array
        Input features: mean FDG-PET in 90 regions, age and sex
        for all subjets of the train set
    y_train : list or np.array
        Amyloid status as assessed with gold-standard methods (ground truth)
        in train set.
    X_train : list or np.array
        Input features: mean FDG-PET in 90 regions, age and sex
        for all subjets of the test set
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    percentile: int or float, optional
        Proportion of most important features to keep. The default is 100.


    Returns
    -------
    best_model : sklearn classifier
        Trained model that achieved best score on validation
        data. If several models achieved best performance,
        final model is the one performing best on test data
    features_test : np.array
        Feature set of test set reduced to indicated percentile.
        If percentiles == 100, full set of features is returned

    """
    # FIND BEST VALIDATION MODEL
    v_ind = np.where(validation == np.max(validation))[0]
    # IF TWO OR MORE VALIDATION MODELS PERFORM BEST, CONSIDER MODEL
    # PERFORMING BEST ON ADNI TEST DATA AS FINAL
    if len(v_ind) > 1:
        max_ind = np.where(np.array(test)[v_ind] == np.max(
            np.array(test)[v_ind]))[0][0]
    else:
        max_ind = 0
    final_ind = np.where(
        (np.array(validation) == validation[v_ind[max_ind]]) &
        (np.array(test) == test[v_ind[max_ind]]))
    best_model = fitted_models[percentile][final_ind[0][0]]
    print("Hyperparameters differing from scikit-learn default:", best_model)

    pickle.dump(best_model,
                open("../results/final_model_{}.p".format(apoe_of_interest),
                     "wb"))

    features_test = transform_data.feature_transform(
        X_train, y_train, X_test, apoe_of_interest, save=True)[1]

    return best_model, features_test


def train_test_adni(df_fdg, df_data, apoe_of_interest, test_size=0.3,
                    upsampling=True, cv=10, scoring='f_tenth', random_state=0,
                    percentiles=[100], info=True):
    """
    Execute cross-validation and testing.

    Training of the gatekeeping system and evaluation on ADNI data.

    Parameters
    ----------
    df_fdg : csv table
        Input features: mean FDG-PET in 90 regions for all available subjects
        (outlier analysis and exclusion must be done prior to this step)
    df_data :  csv table
        Data table including apoe, age and sex information for all subjects.
        Columns must be named PTID for id, APOE4 for apoe, AGE for age
        and PTGENDER for sex.
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25..
        The default is 0.3.
    upsampling : boolean, optional
        Whether or not to use upsampling to avoid overfitting.
        The default is True.
    cv : int, optional
        How many folds to split training data into for cross-validation.
        The default is 10.
    scoring : str or f_tenth
        Should be one out of: "f1", "accuracy", "balanced_accuracy" or f_tenth.
        The default is 'f_tenth'.
    random_state :  int, optional
        random seed. The default is 42.
    percentiles : int or float, optional
        Proportion of most important features to keep. The default is [100].
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

    # GET DATA FROM DATAFRAMES
    subject_vectors, y_true, ids, gender, age = transform_data.extract_data(
        df_fdg, df_data, maj_class=maj_class,
        apoe_of_interest=apoe_of_interest)

    # SPLIT INTO TRAIN AND TEST SET
    X_train, X_test, y_train, y_test, \
        age_train, age_test, ids_train, ids_test, \
        gender_train, gender_test = train_test_split(
            subject_vectors, y_true, age, ids, gender,
            test_size=test_size, stratify=y_true,
            random_state=random_state)
    if info:
        print("TRAIN SET: n_minority class: ", np.bincount(y_train)[0],
              "n_majority class: ", np.bincount(y_train)[1],
              "\nTEST SET: n_minority class: ", np.bincount(y_test)[0],
              "n_majority class: ", np.bincount(y_test)[1])

    # PREPARE DATA
    scaled_data = transform_data.scale_data(
        [X_train, age_train], [X_test, age_test],
        ['X_train', 'age_train'], apoe_of_interest)
    [X_train, age_train] = scaled_data[0]
    [X_test, age_test] = scaled_data[1]

    X_train = np.concatenate(
        (X_train, age_train, np.array(gender_train).reshape(-1, 1)), axis=1)
    X_test = np.concatenate(
        (X_test, age_test, np.array(gender_test).reshape(-1, 1)), axis=1)

    # SET UP MODELS AND LOAD HYPERPARAMETERS
    model_names = ['KNN', 'SVC', 'GPC', 'DNN', 'RFC', 'LOGR']

    model_list = [KNeighborsClassifier(),
                  SVC(random_state=random_state),
                  GaussianProcessClassifier(random_state=random_state,
                                            n_jobs=-1),
                  MLPClassifier(random_state=random_state, max_iter=100),
                  RandomForestClassifier(random_state=random_state),
                  LogisticRegression(random_state=random_state)]
    parameter_space = pickle.load(open("../config/hyperparams_allmodels.p",
                                       "rb"))

    # CROSS-VALIDATED SEARCH FOR OPTIMAL HYPERPARAMETER CONFIGURATIONS
    # yields intermediate models
    v, t, models = classifier_feature_search_cv(
        X_train, y_train,
        X_test, y_test,
        apoe_of_interest=apoe_of_interest,
        maj_class=maj_class,
        upsampling=upsampling,
        cv=cv,
        model_list=model_list, model_names=model_names,
        params=parameter_space,
        percentiles=percentiles,
        scoring=scoring)

    # SHOW RESULTS FROM CROSS-VALIDATION
    df_validation, df_test = show_performance.show_performance_tabular(
        v, t, model_names, percentiles)
    if info:
        show_performance.show_performance_plot(df_validation, df_test,
                                               apoe_of_interest)

    # CHOOSE FINAL MODEL
    best_model, features_test = choose_best_model(
        v, t, models, X_train, y_train, X_test, apoe_of_interest)
    pickle.dump(features_test,
                open("../results/features/ADNI_features_{}.p".format(
                    apoe_of_interest), "wb"))
    pickle.dump(y_test,
                open("../results/features/ADNI_labels_{}.p".format(
                    apoe_of_interest), "wb"))

    # YIELD PREDICTIONS FROM FINAL MODEL FOR FINAL EVAL
    pred = best_model.predict(features_test)
    if info:
        show_performance.show_performance(y_test, pred)

    # show_performance.show_performance_roc(y_test, pred)
    # show_performance.evaluate_misclassif(y_test, pred, maj_class,
    #                                     apoe_of_interest, ids_test)
