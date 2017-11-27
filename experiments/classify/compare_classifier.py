#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:23:00 2017

@author: ajaver
"""
import os
import pandas as pd
import numpy as np
import tables
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
import itertools
import multiprocessing as mp

import time
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix

from compare_ftests import col2ignore

from tqdm import tqdm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cmap=plt.cm.Blues
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    np.set_printoptions(precision=2)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%1.2f' % cm[i, j],
                 horizontalalignment="center",
                 fontsize =12,
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def _get_args(set_type):
    if set_type == 'CeNDR':
        cross_validation_fold = 5
        base_strains = ['N2', 'ED3017', 'CX11314', 'LKC34', 
                          'MY16', 'DL238', 'JT11398', 'JU775',
                          'JU258', 'MY23', 'EG4725', 'CB4856'
                          ]
        main_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/CeNDR/CeNDR_skel_smoothed.hdf5'
        save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/CeNDR/'
        feat_files = {
                'OW' : 'ow_features_CeNDR.csv',
                'tierpsy' :'tierpsy_features_CeNDR.csv'
                }
    elif set_type == 'SWDB':
        cross_validation_fold = 5
        base_strains = ['JU393', 'ED3054', 'JU394', 
                         'N2', 'JU440', 'ED3021', 'ED3017', 
                         'JU438', 'JU298', 'JU345', 'RC301', 
                         'AQ2947', 'ED3049',
                         'LSJ1', 'JU258', 'MY16', 
                         'CB4852', 'CB4856', 'CB4853',
                         ]
        main_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/SWDB/SWDB_skel_smoothed.hdf5'
        save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
        feat_files = {
                'OW_old' : 'ow_features_old_SWDB.csv',
                'OW' : 'ow_features_SWDB.csv',
                'tierpsy' :'tierpsy_features_SWDB.csv'
                }
    return cross_validation_fold, base_strains, main_file, save_dir, feat_files

def _h_cross_validate(args):
    feats_r, col_feats, id_index_str, n_samples, cross_validation_fold, n_estimators = args
    ss = feats_r.groupby('strain').apply(lambda x :x.iloc[random.sample(range(0,len(x)), n_samples)])
    xx = ss[col_feats].values
    
    #center, this shouldn't make that much different in random forest
    #xx = (xx-np.mean(xx, axis=0))/(np.std(xx, axis=0))
    #xx = xx[:, ~np.any(np.isnan(xx), axis=0)]
    
    yy = ss[id_index_str].values
    
    clf = RandomForestClassifier(n_estimators=n_estimators)
    scores = cross_val_score(clf, xx, yy, cv = cross_validation_fold)
    #print(scores)
    return scores



if __name__ == '__main__':
    cross_validation_fold, base_strains, main_file, save_dir, feat_files = _get_args('SWDB')
    
    
    feat_files = {'OW_old':feat_files['OW_old']}
    #%%
    
    with pd.HDFStore(main_file, 'r') as fid:
        df1 = fid['/skeletons_groups']
        df3 = fid['/experiments_data']
        strain_codes = fid['/strains_codes']
    
    cols_to_use = df3.columns.difference(df1.columns)
    df3 = df3[cols_to_use]
    df3 = df3.rename(columns={'id' : 'experiment_id'})
    df = df1.join(df3.set_index('experiment_id'), on='experiment_id')
    #%%
    valid_indices = {}
    with tables.File(main_file, 'r') as fid:
        for set_type in ['train', 'test', 'val']:
            dd = '/index_groups/' + set_type
            if dd in fid:
                v_ind = fid.get_node(dd)[:]
                #use previously calculated indexes to divide data in training, validation and test sets
                valid_indices[set_type] = df.loc[v_ind, 'base_name'].unique()
                
    assert not (set(valid_indices['test']) & set(valid_indices['train']))
    #%%
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, 'F_' + bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        base_strains_id = {x:i for i,x in enumerate(base_strains)}
        feats['strain_base_id'] = feats['strain'].map(base_strains_id)
        
        feats['set_type'] = ''
        for set_type, v_ind in valid_indices.items():
            feats.loc[feats['base_name'].isin(v_ind), 'set_type'] = set_type
        
        feat_data[db_name] = feats
        
    col2ignore_r = col2ignore + ['strain_base_id', 'strain_id', 'set_type']
    #%%
    if False:
        n_trials = 200
        
        n_batch= mp.cpu_count()
        p = mp.Pool(n_batch)
        
        id_index_str = 'strain_base_id'
        
        cross_val_results = {}
        
        start = time.time()
        for db_name, feats in feat_data.items():
            print(db_name)
            
            
            col_feats = [x for x in feats.columns if x not in col2ignore_r]
            good = ~feats['strain_base_id'].isnull() & (feats['set_type'] == 'train')
            feats_r = feats[good]
            
            n_samples = feats_r['strain'].value_counts().min()
            args = feats_r, col_feats, id_index_str, n_samples, cross_validation_fold, 1000
            
            #func = partial(_h_cross_validate, )
            scores = list(p.map(_h_cross_validate, n_trials*[args]))
            scores = np.concatenate(scores)
            #scores = _h_cross_validate(feats_r, col_feats, id_index_str, n_samples, cross_validation_fold)
        
            print("%s Accuracy: %0.2f (+/- %0.2f)" % (db_name, scores.mean(), scores.std() * 2))    
            cross_val_results[db_name] = scores
            
            
            print(time.time() - start)
    
    
    #%%
    if False:
        n_batch= mp.cpu_count()
        p = mp.Pool(n_batch)
        
        id_index_str = 'strain_base_id'
        
        cross_val_results = {}
        
        
        for db_name, feats in feat_data.items():
            print(db_name)
            
            col_feats = [x for x in feats.columns if x not in col2ignore_r]
            good = ~feats['strain_base_id'].isnull() & (feats['set_type'] == 'train')
            feats_r = feats[good]
            
            n_samples = feats_r['strain'].value_counts().min()
            
            
            selected_feats = []
            
            for n_feat in range(30):
                n_trials = 20
                
                all_scores = []
                for iif, ff in enumerate(col_feats):
                    if ff in selected_feats:
                        continue
                    
                    start = time.time()
                    c_feats = selected_feats + [ff]
                    
                    args = feats_r, c_feats, id_index_str, n_samples, cross_validation_fold, 150
                    #func = partial(_h_cross_validate, )
                    scores = list(p.map(_h_cross_validate, n_trials*[args]))
                    scores = np.concatenate(scores)
                    #scores = _h_cross_validate(feats_r, col_feats, id_index_str, n_samples, cross_validation_fold)
                    
                    print(len(c_feats), iif+1, len(col_feats),  c_feats)
                    print("%s Accuracy: %0.2f (+/- %0.2f)" % (db_name, scores.mean(), scores.std() * 2)) 
                    print(scores.min(), scores.max(), scores.mean() - 2*scores.std())
                    all_scores.append((c_feats, scores))
                    
                    print(time.time() - start)
                
                selected_feats = max(all_scores, key=lambda x : x[1].mean() - 2*x[1].std())[0]
                with open('/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/best_feats_OW.txt', 'a+') as fid:
                    fid.write(', '.join(selected_feats) + '\n')
    #%%
    selected_feats = ['midbody_crawling_amplitude_abs', 'foraging_amplitude_abs', 'hips_bend_mean_pos', 'midbody_width_forward', 'neck_bend_sd_forward_abs', 'midbody_speed_pos', 'foraging_speed_pos', 'eigen_projection_4_forward_neg', 'bend_count_backward', 'tail_crawling_frequency_pos', 'length_forward', 'tail_bend_mean_forward_neg', 'head_bend_sd_forward_pos', 'midbody_bend_sd', 'head_bend_sd_neg', 'tail_tip_motion_direction_backward_pos', 'eigen_projection_3_paused_abs', 'head_tip_speed_paused_pos', 'eigen_projection_1_forward_neg', 'eigen_projection_5_forward_pos', 'head_tip_speed_forward', 'tail_speed_forward_neg', 'tail_bend_mean_forward', 'upsilon_turns_time_pos', 'path_range_paused', 'midbody_crawling_frequency', 'head_motion_direction_forward', 'width_length_ratio_backward', 'head_motion_direction_pos', 'path_curvature_paused']
    
    feats = feat_data['OW_old']
    id_index_str = 'strain_base_id'
    
    n_batch= mp.cpu_count()
    p = mp.Pool(n_batch)
    
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    good = ~feats['strain_base_id'].isnull() & (feats['set_type'] == 'train')
    feats_r = feats[good]
    
    n_samples = feats_r['strain'].value_counts().min()
    
    all_scores = []
    for n_feats in range(len(selected_feats)):
        start = time.time()
        feat_cols = selected_feats[:n_feats+1]
        
        print(n_feats, feat_cols)
        n_trials = 200
        args = feats_r, feat_cols, id_index_str, n_samples, cross_validation_fold, 1000
        #func = partial(_h_cross_validate, )
        scores = list(p.map(_h_cross_validate, n_trials*[args]))
        scores = np.concatenate(scores)
        #scores = _h_cross_validate(feats_r, col_feats, id_index_str, n_samples, cross_validation_fold)
    
        print("%s Accuracy: %0.2f (+/- %0.2f)" % (db_name, scores.mean(), scores.std() * 2))    
        all_scores.append(scores)
        print(time.time() - start) 
    #%%
    
    yy = [x.mean() for x in all_scores]
    err = [x.std() for x in all_scores]
    
    plt.figure()
    plt.errorbar(np.arange(1, len(yy)+1), yy, yerr=err)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.savefig('classification_accuracy.png')
    
        
    #%%
    '''
    SWDB
    Random Forest 1000 trees
    200 trials 5-fold cross validation
    
    OW_old Accuracy: 0.57 (+/- 0.16)
    OW Accuracy: 0.52 (+/- 0.16)
    tierpsy Accuracy: 0.52 (+/- 0.17)
    
    
    Accuracy with 30 feats
    
    '''
    
    '''
    CeNDR
    Random Forest 1000 trees
    200 trials 5-fold cross validation
    
    OW Accuracy: 0.77 (+/- 0.13)
    tierpsy Accuracy: 0.71 (+/- 0.13)
    '''
    '''
    
    
    '''
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