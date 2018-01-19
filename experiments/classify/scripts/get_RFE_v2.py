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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from compare_ftests import col2ignore
from sklearn.metrics import f1_score

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
    
    #%% create a dataset with all the features
    feats = feat_data['OW']
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    feats = feats[col_feats + ['base_name']]
    feat_data['all'] = feat_data['tierpsy'].merge(feats, on='base_name')
    
    #%%
    n_estimators = 1000
    n_jobs = 12
    
    results = {}
    for db_name, feats in feat_data.items():
        if db_name != 'all':
            continue
        
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        
        res = []
        sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state=777)
        for ii, (train_index, test_index) in enumerate(sss.split(X, y)):
            #%%
            x_train, y_train  = X[train_index], y[train_index]
            x_test, y_test  = X[test_index], y[test_index]
            
            clf = ExtraTreesClassifier(n_estimators = n_estimators, 
                                       #max_features=128, 
                                       class_weight = 'balanced',
                                       n_jobs = n_jobs,
                                       verbose=1
                                       )
            clf.fit(x_train, y_train)
            
            y_pred_proba = clf.predict_proba(x_test)
            #res.append((y_test, y_pred_proba, clf.feature_importances_.copy()))
            
            y_pred = np.argmax(y_pred_proba, axis=-1)
            
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(ii, f1)
        
        results[db_name] = (res, col_feats)
        
        break