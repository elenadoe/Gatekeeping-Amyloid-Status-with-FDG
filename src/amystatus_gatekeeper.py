#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:50:25 2021

@author: doeringe
"""
import pandas as pd
import warnings
from cv_feature_search import train_test_adni
from testing import test_external, test_balanced
from show_performance import show_importance

warnings.filterwarnings("ignore")


def main(analyze, apoe_of_interest):
    """
    Execute analysis.

    analyze : int
        Which step of the analysis is desired
        1 - Train and test on ADNI data,
        2 - Test on IMC data,
        3 - Test on balanced data,
        4 - Show permutation importance.
    apoe_of_interest : int
        Whether APOE4-nc (0) or APOE4-c (1) are investigated.

    Returns
    -------
    None.

    """
    df_fdg = pd.read_csv('../data/parcels_nooutliers3.csv')
    df_data = pd.read_csv('../data/ADNImerge_amypos_nooutliers_nomask3.csv',
                          sep=";")
    fdg_imc = pd.read_csv("../data/munich_fdg_nomask.csv", sep=";")
    data_imc = pd.read_csv("../data/munich_pib_new_noNA.csv", sep=";")

    if analyze == 1:
        train_test_adni(df_fdg, df_data, apoe_of_interest)
    elif analyze == 2:
        test_external(fdg_imc, data_imc, apoe_of_interest)
    elif analyze == 3:
        print("---ADNI---")
        test_balanced("ADNI", apoe_of_interest)
        print("---IMC---")
        test_balanced("IMC", apoe_of_interest)
    elif analyze == 4:
        # TODO: not tested
        show_importance(apoe_of_interest)
