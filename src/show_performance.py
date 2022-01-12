#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:57:39 2021

@author: doeringe
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import seaborn as sns
from IPython.display import display_html
from sklearn.metrics import roc_curve, auc, fbeta_score, precision_score, \
    recall_score, confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from nilearn.datasets import fetch_atlas_aal
from nilearn import image, plotting
from matplotlib import cm


def show_performance(y_true, pred):
    """
    Print perfomance metrics.

    Parameters
    ----------
    y_true : list or np.ndarray
        DESCRIPTION.
    pred : list
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print("\033[1mPerformance Metrics\033[0m\n",
          "F1/10: {}, Precision: {}, Recall: {}".format(
              np.round(fbeta_score(y_true, pred, beta=1/10), 3),
              np.round(precision_score(y_true, pred), 3),
              np.round(recall_score(y_true, pred), 3)))
    print("\033[1mConfusion Matrix\033[0m\n",
          confusion_matrix(y_true, pred))


def show_performance_tabular(validation, test, model_names, percentiles):
    """
    Show performance of intermediate models in tabular form.

    Intermediate models are the best models from each classifier
    category yielded from cross-validated hyperparameter search.

    Parameters
    ----------
    validation : np.array
        Validation scores.
    t : np.array
        Test scores.
    model_names : list
        List of names of classifiers.
    percentiles : list
        DESCRIPTION.

    Returns
    -------
    df_validation : pd.dataframe
        Validation scores
    df_test : pd.dataframe
        Test scores
    """
    df_validation = pd.DataFrame(
        validation, index=model_names).round(decimals=2)
    df_validation.columns = percentiles
    dv_style = df_validation.style.set_table_attributes(
        "style='display:inline'").set_caption("Cross-validation Results")

    df_test = pd.DataFrame(
        test, index=model_names).round(decimals=2)
    df_test.columns = percentiles
    dt_style = df_test.style.set_table_attributes("style='display:inline'").\
        set_caption("Test Results")

    display_html(dv_style._repr_html_()+dt_style._repr_html_(), raw=True)
    return df_validation, df_test


def show_performance_plot(df_validation, df_test, apoe):
    """
    Show performance of intermediate models in bar plots.

    Intermediate models are the best models from each classifier
    category yielded from cross-validated hyperparameter search.

    Parameters
    ----------
    df_validation : pd.dataframe
        Validation scores
    df_test : pd.dataframe
        Test scores
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    df_validation.transpose().plot(kind='bar', colormap='cividis', ax=ax[0])
    ax[0].set_ylabel(r'F$\beta$ Score')
    ax[0].set_xlabel('Percentile of Mutual Information [%]')
    ax[0].set_title('Best Model Performance (Validation)')
    ax[0].set_ylim(0.50, 1.0)
    ax[0].get_legend().remove()

    df_test.transpose().plot(kind='bar', colormap='cividis', ax=ax[1])
    ax[1].set_ylabel(r'F$\beta$ Score')
    ax[1].set_xlabel('Percentile of Mutual Information [%]')
    ax[1].set_title('Best Model Performance (Test)')
    ax[1].set_ylim(0.50, 1.0)

    plt.legend(bbox_to_anchor=[1.02, 1], loc='upper left')
    plt.savefig('../results/Gatekeeping_{}.jpg'.format(str(apoe)),
                bbox_inches='tight')
    plt.show()


def show_performance_roc(y_true, pred):
    """


    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    pred : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fpr, tpr, threshold = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label="AUC = %0.2f" % roc_auc, color="red")
    plt.plot([0, 1], [0, 1], '--', color="black", label="Chance = %0.2f" % 0.5)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


def evaluate_misclassif(y_true, pred, pos_label, apoe_of_interest, ids_test):
    """


    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    pred : TYPE
        DESCRIPTION.
    pos_label : TYPE
        DESCRIPTION.
    apoe : TYPE
        DESCRIPTION.
    ids_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    neg_label = np.abs(pos_label-1)
    class_ = ['tn' if (y_true[i] == pred[i] == neg_label)
              else 'tp' if (y_true[i] == pred[i] == pos_label)
              else 'fn' if (y_true[i] == pos_label and pred[i] == neg_label)
              else 'fp' if (y_true[i] == neg_label and pred[i] == pos_label)
              else 'error' for i in range(len(y_true))]
    correctness = pd.concat((pd.DataFrame(ids_test, columns=['ID']),
                             pd.DataFrame(y_true, columns=['GT']),
                             pd.DataFrame(pred, columns=['PRED']),
                             pd.DataFrame(class_, columns=['CLASSIF'])),
                            axis=1)

    correctness.to_csv("../results/correctness_{}.csv".format(
        apoe_of_interest))


def show_signal_dist(features_test_raw, pred, names, l_):
    """
    # TODO: include into permutation importance.

    Parameters
    ----------
    features_test_raw : TYPE
        DESCRIPTION.
    pred : TYPE
        DESCRIPTION.
    names : TYPE
        DESCRIPTION.
    l_ : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sns.set(font_scale=1.3)
    sns.set_style("dark")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    reg_ind = np.argmax(l_)[0]
    sns.violinplot(y=np.array(features_test_raw)[:, reg_ind],
                   x=pred, ax=ax[0],
                   palette="Paired")
    ax[0].set_ylabel(np.array(names)[reg_ind])
    ax[0].set_xlabel('Prediction')

    sns.violinplot(y=np.array(features_test_raw)[:, 90], x=pred, ax=ax[1],
                   palette="Paired")
    ax[1].set_ylabel("Age")
    ax[1].set_xlabel('Prediction')

    plt.savefig("../../results/regions_apoe-nc_violin_topfeat.jpg")
    plt.show()


def show_importance(apoe_of_interest, repetitions=1000, random_state=0,
                    info=True):
    """
    Calculate and show permutation importance.

    Takes trained classifier and saved test predictions of the ADNI
    dataset to calculate permutation importance of each input feature.
    More about permutation importance:
    https://scikit-learn.org/stable/modules/permutation_importance.html

    Parameters
    ----------
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.
        Must be between 0 - 2.
    repetitions : int, optional
        Number of times to permute a feature. The default is 1000.
    random_state : int, optional
        Pseudo-random number generator to control the permutations
        of each feature. Pass an int to get reproducible results
        across function calls.  The default is 0.
    info : boolean, optional
        Whether to print information. The default is True.

    Returns
    -------
    None.

    """
    # LOAD FEATURES, LABELS, MODEL, FEATURE NAMES AND FEATURE SELECTOR
    features = pickle.load(
        open("../results/features/ADNI_features_{}.p".format(
            apoe_of_interest), "rb"))
    y_true = pickle.load(
        open("../results/features/ADNI_labels_{}.p".format(
            apoe_of_interest), "rb"))
    model = pickle.load(
        open("../results/final_model_{}.p".format(
            apoe_of_interest), "rb"))
    names = pickle.load(open("../config/region_names.p", "rb")) +\
        ['age', 'gender']
    feature_select = pickle.load(open("../config/feature_select_{}.p".format(
        apoe_of_interest), "rb"))
    feature_ind = feature_select.get_support(indices=True)
    feature_names = np.array(names)[feature_ind]

    # CALCULATE PERMUTATION IMPORTANCE
    feature_importance = permutation_importance(
        model, features, y_true,
        scoring=make_scorer(fbeta_score, beta=1/10),
        n_repeats=repetitions, random_state=random_state)
    feat_imp = feature_importance.importances_mean

    # SAVE PERMUTATION IMPORTANCE OF EACH FEATURE,
    # SET PERMUTATION IMPORTANCE OF NON-ASSESSED FEATURES TO ZERO
    # (NON-CEREBELLAR REGIONS WERE NOT ASSESSED, ADDITIONAL REGIONS
    # DON'T HAVE PERMUTATION IMPORTANCE IF FEATURE SELECTION
    # WAS PREVIOUSLY APPLIED)
    l_ = [0]*118
    count = 0
    txt = open("../results/permutation_importance_{}.txt".format(
        apoe_of_interest), "w")
    for n in range(118):
        if n in feature_ind:
            txt.write(str(n) + "\t" + feature_names[n] + "\t" +
                      str(round(feat_imp[count], 3)))
            l_[n] = feat_imp[count]
            count += 1
        else:
            l_[n] = 0

    # LOAD ATLAS FOR PLOTTING
    aal = fetch_atlas_aal()
    atlas = image.load_img(aal.maps)
    atlas_matrix = image.get_data(atlas)

    # FOR PLOTTING OF BRAIN REGION IMPORTANCE, SET PERMUTATION
    # IMPORTANCE AT POSITION 90 AND 91 (CORRESPONDING TO AGE AND SEX)
    # TO 0
    l_atlas = l_
    l_atlas[90] = 0
    l_atlas[91] = 0

    # CREATE STATISTICAL MAP WHERE EACH VOXEL VALUE CORRESPONDS
    # TO PERMUTATION IMPORTANCE OF THIS BRAIN REGION
    atlas_matrix_stat = atlas_matrix.copy()
    aal_numbers = np.unique(atlas_matrix_stat).tolist()
    for x in range(117):
        if x == 0:
            pass
        else:
            curr_x = aal_numbers[x]
            atlas_matrix_stat = np.where(
                atlas_matrix_stat == curr_x, l_atlas[x-1], atlas_matrix_stat)

    # SAVE RESULT AS NIFTI IMAGE
    atlas_final = image.new_img_like(atlas, atlas_matrix_stat)
    nib.save(atlas_final,
             "../results/permutation_importance_{}.nii".format(
                 apoe_of_interest))
    if info:
        print("Total permutation importance: ",
              sum(feature_importance.importances_mean))
        print("Total positive permutation importance: ",
              sum(feature_importance.importances_mean[
                  np.where(feature_importance.importances_mean > 0)]))
        print("Text file and Nifti for permutation importance ",
              "stored under ../results/permutation_importance_{}".format(
                  apoe_of_interest))

        fig = plt.figure(figsize=(10, 5), frameon=False)

        plotting.plot_stat_map(atlas_final,
                               cmap=cm.get_cmap("viridis"), draw_cross=False,
                               figure=fig, black_bg=True,
                               cut_coords=(35, -37, 15))
        # plt.savefig("../results/knn_regions_apoe-c_multim.jpg")
        plt.show()
