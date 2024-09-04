# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:24:49 2024

@author: ayhan
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def json_to_df():

def preprocess_data(df):
    #apply timestamps
    
    #apply bins
    
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    
    #standardize the data
    