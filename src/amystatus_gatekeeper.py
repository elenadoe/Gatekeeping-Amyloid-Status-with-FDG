#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:50:25 2021

@author: doeringe
"""
import pandas as pd
import warnings
from cv_feature_search import train_test_adni
from testing import test_external, test_balanced, show_mlp_loss
from show_performance import show_importance, show_signal_dist

warnings.filterwarnings("ignore")

random_state = 42

def main(analyze, apoe_of_interest, try_="",
         drop_features=None, percentile = 100, info=True):
    """
    Execute analysis.

    analyze : int
        Which step of the analysis is desired
        1 - Train and test on ADNI data,
        2 - Test on IMC data,
        3 - Test on balanced data,
        4 - show loss curve
        5 - Show permutation importance.
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.

    Returns
    -------
    None.

    """
    df_fdg = pd.read_csv('../data/ADNI_parcels_onlypet_rev1.csv')
    # df_fdg = pd.read_csv('../data/ADNI_parcels_rev1.csv')
    #df_data = pd.read_csv(
    #    '../data/ADNI_merge_nooutliers_rev1.csv')
    df_data = pd.read_csv(
        '../data/ADNI_merge_nooutliers_onlypet_rev1.csv')
    fdg_imc = pd.read_csv("../data/Munich_parcels.csv", sep=";")
    data_imc = pd.read_csv("../data/munich_pib_new_noNA.csv")

    if analyze == 1:
        train_test_adni(df_fdg, df_data, apoe_of_interest,
                        drop_features=drop_features, info=info,
                        percentile=percentile, try_=try_)
    elif analyze == 2:
        test_external(fdg_imc, data_imc, apoe_of_interest,
                      drop_features=drop_features,
                      percentile=percentile, try_=try_)
    elif analyze == 3:
        print("---ADNI---")
        test_balanced("ADNI", apoe_of_interest, percentile=percentile,
                      drop_features=drop_features,
                      random_state=random_state, try_=try_)
        print("---IMC---")
        test_balanced("IMC", apoe_of_interest, percentile=percentile,
                      drop_features=drop_features,
                      random_state=random_state, try_=try_)
    elif analyze == 4:
        show_mlp_loss(apoe_of_interest, percentile,
                      drop_features=drop_features, try_=try_)
    elif analyze == 5:
        pred_csv = pd.read_csv(
            "../results/Predictions_ADNImerge_{}_{}_wo-{}{}.csv".format(
                percentile, apoe_of_interest, drop_features, try_))
        show_importance(apoe_of_interest, pred_csv, data_fdg=df_fdg,
                        percentile=percentile,
                        drop_features=drop_features, try_=try_)
