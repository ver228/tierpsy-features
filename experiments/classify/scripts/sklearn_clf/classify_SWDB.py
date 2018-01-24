#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:23:00 2017

@author: ajaver
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
import pickle
import itertools
import multiprocessing as mp

import time
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, f1_score
from compare_ftests import col2ignore

from tqdm import tqdm
#%%


if __name__ == '__main__':
    
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../data/SWDB'
    feat_files = {
            'OW' : 'F0.025_ow_features_old_SWDB.csv',
            'tierpsy' :'F0.025_tierpsy_features_SWDB.csv'
            }
    
    
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        #maybe i should divided it in train and test, but cross validation should be enough...
        feats['set_type'] = ''
        
        feat_data[db_name] = feats
        
    col2ignore_r = col2ignore + ['strain_id', 'set_type']
    #%% scale data
    for db_name, feats in feat_data.items(): 
        col_val = [x for x in feats.columns if x not in col2ignore_r]
        
        dd = feats[col_val]
        z = (dd-dd.mean())/(dd.std())
        feats[col_val] = z
        feat_data[db_name] = feats
    #%%
    n_estimators = 1000
    n_jobs = 20
    
    results = {}
    for db_name, feats in feat_data.items():
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
            
            #clf = ExtraTreesClassifier(n_estimators = n_estimators, 
            #                           class_weight = 'balanced',
            #                           n_jobs = n_jobs,
            #                           verbose=1)
            
            clf = SVC(C= 1.,
                      kernel = 'rbf',
                      probability = True,
                      class_weight = 'balanced',
                      decision_function_shape = 'ovr',
                      verbose=1)
            
            
            #clf = LogisticRegression(C = 1., 
            #                         #multi_class = 'multinomial',
            #                           class_weight = 'balanced',
            #                           n_jobs = n_jobs,
            #                           solver = 'saga',
            #                           verbose=1)
            
            
            clf.fit(x_train, y_train)
            
            y_pred_proba = clf.predict_proba(x_test)
            res.append((y_test, y_pred_proba, clf))
            
            y_pred = np.argmax(y_pred_proba, axis=-1)
            
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(ii, f1)
        
        results[db_name] = (res, col_feats)
        
        break
 