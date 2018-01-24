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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
            
            clf = RandomForestClassifier(n_estimators = n_estimators, 
                                       criterion = 'gini',
                                       class_weight = 'balanced',
                                       n_jobs = n_jobs,
                                       verbose=1)
            clf.fit(x_train, y_train)
            
            y_pred_proba = clf.predict_proba(x_test)
            res.append((y_test, y_pred_proba, clf.feature_importances_.copy()))
            
            y_pred = np.argmax(y_pred_proba, axis=-1)
            
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(db_name, ii, f1)
        
        results[db_name] = (res, col_feats)
    #%%
    save_model_name = 'RF_class_results_SB.pkl'
    #%%
    with open(save_model_name, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
    #%%
    with open(save_model_name, 'rb') as f:
        results = pickle.load(f)
    
    #%%
    strains = np.sort(feats['strain'].unique())
    
    for db_name, (res, col_feats) in results.items():
        y_test, y_pred_proba, f_importances = map(np.array, list(zip(*res)))
        
        ind_s = np.argsort(y_pred_proba, axis=-1)
        
        _, _, real_t = np.where(ind_s == y_test[..., None])
        real_t = 357 - real_t
    
        print(db_name)
        for tt in [1, 2, 3, 5, 10, 20, 50, 100]:
            print('top {} : {}'.format(tt, np.mean(real_t<tt)))
    #%%
    res_sum = {}
    for db_name, (res,col_feats) in results.items():
        y_test, y_pred_proba, f_importances = map(np.array, list(zip(*res)))
        f_importances_avg = np.mean(f_importances, axis=0)
        f_importances_avg = pd.Series(f_importances_avg, index= col_feats)
        f_importances_avg = f_importances_avg.sort_values()
        
        y_pred = np.argmax(y_pred_proba, axis=-1)
        cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
        
        tp = np.diag(cm)
        tp_fp = cm.sum(axis=1) + 1e-10
        tp_fn = cm.sum(axis=0) + 1e-10
        
        precision = tp/tp_fp
        recall = tp/tp_fn
        F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        stat = (precision, recall, F1)
        
        res_sum[db_name] = (f_importances_avg, cm, stat)
        
        
        cm_n = cm/(cm.sum(axis=1) + 1e-10)
    #%%
    plt.figure()
    l_h = []
    for k, val in res_sum.items():
        y = np.sort(val[-1][-1])
    
        h, = plt.plot(y, label = k)
        l_h.append(h)
    
    plt.legend(handles = l_h)
    plt.title('Sorted f1-scores')
    #%%
    y_tierpsy = res_sum['tierpsy'][0]
    y_OW = res_sum['OW'][0]
    
    plt.figure()
    l_h = []
    for k, val in res_sum.items():
        val = val[0]
        yy = val.values*val.size
        
        xx = np.linspace(0, 1, val.size)
        
        h, = plt.plot(xx, yy, label = k)
        #h, = plt.plot(yy, label = k)
        l_h.append(h)
    
    plt.legend(handles = l_h)
    plt.title('Feature Importances')
    #%%
#    cross_validation_fold = 5
#    n_trials = 200
#    
#    n_batch= mp.cpu_count()
#    p = mp.Pool(n_batch)
#    
#    id_index_str = 'strain_id'
#    
#    cross_val_results = {}
#    
#    start = time.time()
#    for db_name, feats in feat_data.items():
#        print(db_name)
#        
#        
#        col_feats = [x for x in feats.columns if x not in col2ignore_r]
#        
#        n_samples = feats['strain'].value_counts().min()
#        args = feats, col_feats, id_index_str, n_samples, cross_validation_fold, 1000
#        
#        #func = partial(_h_cross_validate, )
#        scores = list(p.map(_h_cross_validate, n_trials*[args]))
#        scores = np.concatenate(scores)
#        #scores = _h_cross_validate(feats_r, col_feats, id_index_str, n_samples, cross_validation_fold)
#    
#        print("%s Accuracy: %0.2f (+/- %0.2f)" % (db_name, scores.mean(), scores.std() * 2))    
#        cross_val_results[db_name] = scores
#        
#        
#        print(time.time() - start)
#    
#    
#    #%%
#    if False:
#        n_batch= mp.cpu_count()
#        p = mp.Pool(n_batch)
#        
#        id_index_str = 'strain_base_id'
#        
#        cross_val_results = {}
#        
#        
#        for db_name, feats in feat_data.items():
#            print(db_name)
#            
#            col_feats = [x for x in feats.columns if x not in col2ignore_r]
#            good = ~feats['strain_base_id'].isnull() & (feats['set_type'] == 'train')
#            feats_r = feats[good]
#            
#            n_samples = feats_r['strain'].value_counts().min()
#            
#            
#            selected_feats = []
#            
#            for n_feat in range(30):
#                n_trials = 20
#                
#                all_scores = []
#                for iif, ff in enumerate(col_feats):
#                    if ff in selected_feats:
#                        continue
#                    
#                    start = time.time()
#                    c_feats = selected_feats + [ff]
#                    
#                    args = feats_r, c_feats, id_index_str, n_samples, cross_validation_fold, 150
#                    #func = partial(_h_cross_validate, )
#                    scores = list(p.map(_h_cross_validate, n_trials*[args]))
#                    scores = np.concatenate(scores)
#                    #scores = _h_cross_validate(feats_r, col_feats, id_index_str, n_samples, cross_validation_fold)
#                    
#                    print(len(c_feats), iif+1, len(col_feats),  c_feats)
#                    print("%s Accuracy: %0.2f (+/- %0.2f)" % (db_name, scores.mean(), scores.std() * 2)) 
#                    print(scores.min(), scores.max(), scores.mean() - 2*scores.std())
#                    all_scores.append((c_feats, scores))
#                    
#                    print(time.time() - start)
#                
#                selected_feats = max(all_scores, key=lambda x : x[1].mean() - 2*x[1].std())[0]
#                with open('/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/best_feats_OW.txt', 'a+') as fid:
#                    fid.write(', '.join(selected_feats) + '\n')
#    #%%
#    selected_feats = ['midbody_crawling_amplitude_abs', 'foraging_amplitude_abs', 'hips_bend_mean_pos', 'midbody_width_forward', 'neck_bend_sd_forward_abs', 'midbody_speed_pos', 'foraging_speed_pos', 'eigen_projection_4_forward_neg', 'bend_count_backward', 'tail_crawling_frequency_pos', 'length_forward', 'tail_bend_mean_forward_neg', 'head_bend_sd_forward_pos', 'midbody_bend_sd', 'head_bend_sd_neg', 'tail_tip_motion_direction_backward_pos', 'eigen_projection_3_paused_abs', 'head_tip_speed_paused_pos', 'eigen_projection_1_forward_neg', 'eigen_projection_5_forward_pos', 'head_tip_speed_forward', 'tail_speed_forward_neg', 'tail_bend_mean_forward', 'upsilon_turns_time_pos', 'path_range_paused', 'midbody_crawling_frequency', 'head_motion_direction_forward', 'width_length_ratio_backward', 'head_motion_direction_pos', 'path_curvature_paused']
#    
#    feats = feat_data['OW_old']
#    id_index_str = 'strain_base_id'
#    
#    n_batch= mp.cpu_count()
#    p = mp.Pool(n_batch)
#    
#    col_feats = [x for x in feats.columns if x not in col2ignore_r]
#    good = ~feats['strain_base_id'].isnull() & (feats['set_type'] == 'train')
#    feats_r = feats[good]
#    
#    n_samples = feats_r['strain'].value_counts().min()
#    
#    all_scores = []
#    for n_feats in range(len(selected_feats)):
#        start = time.time()
#        feat_cols = selected_feats[:n_feats+1]
#        
#        print(n_feats, feat_cols)
#        n_trials = 200
#        args = feats_r, feat_cols, id_index_str, n_samples, cross_validation_fold, 1000
#        #func = partial(_h_cross_validate, )
#        scores = list(p.map(_h_cross_validate, n_trials*[args]))
#        scores = np.concatenate(scores)
#        #scores = _h_cross_validate(feats_r, col_feats, id_index_str, n_samples, cross_validation_fold)
#    
#        print("%s Accuracy: %0.2f (+/- %0.2f)" % (db_name, scores.mean(), scores.std() * 2))    
#        all_scores.append(scores)
#        print(time.time() - start) 
#    #%%
#    
#    yy = [x.mean() for x in all_scores]
#    err = [x.std() for x in all_scores]
#    
#    plt.figure()
#    plt.errorbar(np.arange(1, len(yy)+1), yy, yerr=err)
#    plt.ylabel('Accuracy')
#    plt.xlabel('Number of features')
#    plt.savefig('classification_accuracy.png')
#    
#        
#    #%%
#    '''
#    SWDB
#    Random Forest 1000 trees
#    200 trials 5-fold cross validation
#    
#    OW_old Accuracy: 0.57 (+/- 0.16)
#    OW Accuracy: 0.52 (+/- 0.16)
#    tierpsy Accuracy: 0.52 (+/- 0.17)
#    
#    
#    Accuracy with 30 feats
#    
#    '''
#    
#    '''
#    CeNDR
#    Random Forest 1000 trees
#    200 trials 5-fold cross validation
#    
#    OW Accuracy: 0.77 (+/- 0.13)
#    tierpsy Accuracy: 0.71 (+/- 0.13)
#    '''
    #%%
#    id_index_str = 'strain_base_id'
#    classifiers_d = {}
#    for db_name, datasets  in sets2test.items():
#        col_feats = [x for x in datasets['train'].columns if x not in col2ignore_r]
#        
#        ss = datasets['train'].dropna(subset=id_index_str)
#        X_train = ss[col_feats].values
#        Y_train = ss[id_index_str].values
#        print(db_name, ss.shape)
#        
#        ss = datasets['test'].dropna(subset=id_index_str)
#        x_test = ss[col_feats].values
#        y_test = ss[id_index_str].values
#        print(db_name, ss.shape)
#        
#        clf = RandomForestClassifier(n_estimators=1000)
#        clf.fit(X_train, Y_train)
#    
#        proba = clf.predict_proba(x_test)
#        top_pred = np.argsort(proba, axis=1)[: ,::-1]
#        preds = top_pred==y_test[:, np.newaxis]
#        print(db_name, preds[:,0].mean())
#        
#        classifiers_d[db_name] = clf
#        #%%
#        dr = {v:k for k,v in base_strains_id.items()}
#        
#        yt = [dr[x] for x in y_test]
#        yp = [dr[x] for x in top_pred[:, 0]]
#        labels = sorted(dr.values())
#        cm = confusion_matrix(yt, yp, labels=labels)
#        
#        plt.figure(figsize=(21,21))
#        plot_confusion_matrix(cm, labels, normalize = True)
#        plt.title(db_name)
#    #%%
#    classifiers_d = {}
#    for db_name, datasets  in sets2test.items():
#        print(db_name, datasets['train'].shape)
#        
#        col_feats = [x for x in datasets['train'].columns if x not in col2ignore]
#        
#        X_train = datasets['train'][col_feats].values
#        Y_train = datasets['train']['strain_id'].values
#        
#        x_test = datasets['test'][col_feats].values
#        y_test = datasets['test']['strain_id'].values
#        
#        clf = RandomForestClassifier(n_estimators=1000)
#        clf.fit(X_train, Y_train)
#        
#        proba = clf.predict_proba(x_test)
#        top_pred = np.argsort(proba, axis=1)[: ,::-1]
#        preds = top_pred==y_test[:, np.newaxis]
#        print(db_name, preds[:,0].mean())
#        
#        
#        classifiers_d[db_name] = clf
#    
#    #%%
#    for db_name, bn in feat_files.items():
#        print(db_name)
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