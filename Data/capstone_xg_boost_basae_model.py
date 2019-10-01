# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:14:01 2019

@author: nsmk6
"""

import pandas as pd
col = ['sepsis_angus','sepsis_martin','sepsis_explicit','septic_shock_explicit','severe_sepsis_explicit','sepsis_nqf','sepsis_cdc','sepsis_cdc_simple']

df = pd.read_csv('Complete_sepsis3.csv')

def conditions(s):
    if ((s['sepsis_angus'] == 1) or (s['sepsis_martin'] == 1) or (s['sepsis_explicit'] == 1) or (s['septic_shock_explicit'] == 1) or (s['severe_sepsis_explicit'] == 1) or (s['sepsis_nqf'] == 1) or (s['sepsis_cdc'] == 1) or (s['sepsis_cdc_simple'] == 1)):
        return 1
    else:
        return 0 

df['sepsis_3'] = df.apply(conditions, axis=1)

df['antibiotic_time_poe'] = pd.to_datetime(df['antibiotic_time_poe'], format='%Y-%m-%d %H:%M:%S')
df['blood_culture_time'] = pd.to_datetime(df['blood_culture_time'], format='%Y-%m-%d %H:%M:%S')

df['time_diff_1'] = (df.antibiotic_time_poe-df.blood_culture_time).astype('timedelta64[D]')
df['time_diff_2'] = (df.blood_culture_time-df.antibiotic_time_poe).astype('timedelta64[D]')

def time_diff(s):
    if s.time_diff_1 > 0:
        return s.time_diff_1
    elif s.time_diff_2 > 0:
        return s.time_diff_2
    else:
        return 0

df['time'] = df.apply(time_diff, axis=1)

df.drop(columns=['time_diff_1', 'time_diff_2'], inplace=True)

df.dtypes

df = df.select_dtypes(exclude=['object'])
df.drop(columns=col, inplace=True)
df.drop(columns=['icustay_id','hadm_id','antibiotic_time_poe','blood_culture_time'], inplace=True)


X = df.loc[:, df.columns != 'sepsis_3'].values
y = df.sepsis_3.values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
