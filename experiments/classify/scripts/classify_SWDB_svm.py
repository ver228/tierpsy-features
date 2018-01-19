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
from sklearn.ensemble import ExtraTreesClassifier #RandomForestClassifier
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
        
#    #%%
#    save_model_name = 'RF_class_results_SB.pkl'
#    #%%
#    with open(save_model_name, 'wb') as f:
#        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
#    
#    #%%
#    with open(save_model_name, 'rb') as f:
#        results = pickle.load(f)
#    
#    #%%
#    strains = np.sort(feats['strain'].unique())
#    
#    for db_name, (res, col_feats) in results.items():
#        y_test, y_pred_proba, f_importances = map(np.array, list(zip(*res)))
#        
#        ind_s = np.argsort(y_pred_proba, axis=-1)
#        
#        _, _, real_t = np.where(ind_s == y_test[..., None])
#        real_t = 357 - real_t
#    
#        print(db_name)
#        for tt in [1, 2, 3, 5, 10, 20, 50, 100]:
#            print('top {} : {}'.format(tt, np.mean(real_t<tt)))
#    #%%
#    res_sum = {}
#    for db_name, (res,col_feats) in results.items():
#        y_test, y_pred_proba, f_importances = map(np.array, list(zip(*res)))
#        f_importances_avg = np.mean(f_importances, axis=0)
#        f_importances_avg = pd.Series(f_importances_avg, index= col_feats)
#        f_importances_avg = f_importances_avg.sort_values()
#        
#        y_pred = np.argmax(y_pred_proba, axis=-1)
#        cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
#        
#        tp = np.diag(cm)
#        tp_fp = cm.sum(axis=1) + 1e-10
#        tp_fn = cm.sum(axis=0) + 1e-10
#        
#        precision = tp/tp_fp
#        recall = tp/tp_fn
#        F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
#        
#        stat = (precision, recall, F1)
#        
#        res_sum[db_name] = (f_importances_avg, cm, stat)
#    #%%
#    y_tierpsy = res_sum['tierpsy'][-1][-1]
#    y_OW = res_sum['OW'][-1][-1]
#    
#    ind_t = np.argsort(y_tierpsy)
#    ind_ow = np.argsort(y_OW)
#    
#    h_ow, = plt.plot(y_OW[ind_ow], label = 'OW')
#    h_ti, = plt.plot(y_tierpsy[ind_t], label = 'tierpsy')
#    
#    
#    plt.legend(handles = [h_ow, h_ti])
#    
#    #%%
#    y_tierpsy = res_sum['tierpsy'][0]
#    y_OW = res_sum['OW'][0]
#    
#    plt.figure()
#    l_h = []
#    for k, val in res_sum.items():
#        val = val[0]
#        yy = val.values#*val.size
#        
#        xx = np.linspace(0, 1, val.size)
#        
#        #h, = plt.plot(xx, yy, label = k)
#        h, = plt.plot(yy, label = k)
#        l_h.append(h)
#    
#    plt.legend(handles = l_h)
#    plt.title('Feature Importances')
#        fname = os.path.join(save_dir, 'F_' + bn)
#        feats = pd.read_csv(fname)
#    
#        col_feats = [x for x in feats.columns if x not in col2ignore]
#        
#        clf = RandomForestClassifier(n_estimators=1000)
#        xx = feats[col_feats].values
#        
#        
#        ss = np.sort(feats['strain'].unique())
#        s_dict = {s:ii for ii,s in enumerate(ss)}
#        feats['strain_id'] = feats['strain'].map(s_dict)
#        yy = feats['strain_id'].values
#        
#        scores = cross_val_score(clf, xx, yy, cv = cross_validation_fold)
#    
#        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))