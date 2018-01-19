#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:34:56 2018

@author: ajaver
"""
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from compare_ftests import col2ignore

if __name__ == '__main__':
    
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../data/SWDB'
    feat_files = {
            'OW' : 'F0.025_ow_features_old_SWDB.csv',
            #'OW' : 'ow_features_SWDB.csv',
            'tierpsy' :'F0.025_tierpsy_features_SWDB.csv'
            }
    #%%
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        feat_data[db_name] = feats
        
    col2ignore_r = col2ignore + ['strain_id', 'set_type']
    #%%
    n_jobs = 6
    n_estimators = 10
    
    results = {}
    for db_name, feats in feat_data.items():
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
        clf = ExtraTreesClassifier(n_estimators = n_estimators, 
                                           #max_features=128, 
                                           class_weight = 'balanced',
                                           n_jobs = n_jobs
                                           )  
        
        selector = RFECV(clf, step=1, cv=10, verbose=2)
        selector = selector.fit(X, y)
        
        results[db_name] = selector