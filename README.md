# Gatekeeping-Amyloid-Status-with-FDG

A demonstration of the analyses can be found under src/demo.ipynb.

**BACKGROUND**

In patients with mild cognitive impairment (MCI), enhanced cerebral amyloid-β plaque burden is a high-risk factor to develop dementia with Alzheimer’s disease (AD). Not all patients have immediate access to the assessment of amyloid status (A-status) via gold standard methods. It may therefore be of interest to find suitable biomarkers to preselect patients benefitting most from additional workup of the A-status. In this study, we propose a machine learning-based gatekeeping system for the prediction of A-status on the grounds of pre-existing information on APOE-genotype 18F-FDG PET, age and sex.

**METHODS**

Four hundred and ten MCI patients were used to train different machine learning classifiers to predict amyloid status majority classes among APOE-ε4 non-carriers (APOE4-nc; majority class: amyloid negative (Aβ-)) and carriers (APOE4-c; majority class: amyloid positive (Aβ+)) from 18F-FDG-PET, age and sex. Classifiers were tested on two different datasets. Finally, frequencies of conversion to AD were compared between gold standard and predicted amyloid status.

![Alt text](https://github.com/elenadoe/Gatekeeping-Amyloid-Status-with-FDG/blob/main/FigS1.png "Classification Pipeline")

**RESULTS**

Aβ- was predicted with a precision of 81 – 83% in APOE4-nc and Aβ+ was predicted with 86 – 90% precision in APOE4-c, respectively. Predicted and gold standard A-status were equally indicative of risk of conversion to AD.

**CONCLUSION**

We have developed a gatekeeping methodology allowing approximation of amyloid status in MCI with good reliability using APOE-genotype, 18F-FDG PET, age and sex information. The algorithm could enable better estimation of individual risk for developing AD based on existing biomarker information, and support efficient selection of patients who would benefit most from further etiological clarification. Further potential utility in clinical routine and clinical trials is discussed.

![Alt text](https://github.com/elenadoe/Gatekeeping-Amyloid-Status-with-FDG/blob/main/Fig2.jpg "Clinical Utility of the Gatekeeping Methodology")

If you make use of (parts of) the code used in this project, please cite us.
The paper is currently under revision at EJNMMM.
