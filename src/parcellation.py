#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:23:26 2022

@author: doeringe
"""

from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from nilearn.datasets import fetch_atlas_aal

modality = "fdg"

image_list = []
subj_succ = {}
subj_succ['PTID'] = []

# create list of regional data and subject IDs
if modality == "fdg":

    # subjs = pd.read_csv('../data/ADNI_merge_Amyloid_4.csv', sep=";")
    subjs = pd.read_csv('../data/munich_pib_new_noNA.csv')
    subjs_list = subjs['PTID'].tolist()
    # data_path = '../../SUVR/'
    data_path = '../../Test_Munich_Data2/FDG_Scans/SUVR_new2/'

    atlas = fetch_atlas_aal()
    labels = atlas.labels()

    output_csv = '../data/Munich_parcels2.csv'

    for sub in subjs_list:
        foi = glob(data_path + "SUVr*" + sub + "*")
        if foi:
            this_image = nib.load(foi[0])
            niimg = check_niimg(this_image, atleast_4d=True)
            masker = NiftiLabelsMasker(labels_img=atlas.maps,
                                       standardize=False,
                                       memory='nilearn_cache',
                                       resampling_target='data')
            parcelled = masker.fit_transform(niimg)
            image_list.append(parcelled)
            subj_succ['PTID'].append(sub)
elif modality == "pib":

    subjs = pd.read_csv('../data/Munich_dem.csv', sep=";")
    subjs_list = subjs['PTID'].tolist()
    data_path = '../../Test_Munich_Data2/PIB_Scans/SUVR_new/'
    pib_mask = '../0_templates/pib_mask_global.nii'

    output_csv = '../data/Munich_pib.csv'
    labels = ['pib_global']

    for sub in subjs_list:
        foi = glob(data_path + "SUVr*" + sub + "*")
        print(foi)
        if foi:
            this_image = nib.load(foi[0])
            niimg = check_niimg(this_image, atleast_4d=True)
            masker = NiftiLabelsMasker(labels_img=pib_mask,
                                       standardize=False,
                                       memory='nilearn_cache',
                                       resampling_target='data')
            parcelled = masker.fit_transform(niimg)
            image_list.append(parcelled)
            subj_succ['PTID'].append(sub)

features = np.array(image_list)
x, y, z = features.shape
features = features.reshape(x, z)
df = pd.DataFrame(features, columns=labels)
df_sub = pd.DataFrame(subj_succ)
df_final = pd.concat([df_sub, df], axis=1)

df_final.to_csv(output_csv, index=False)
