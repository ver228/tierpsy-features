#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:23:00 2017

@author: ajaver
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
import multiprocessing as mp

import time
import matplotlib.pylab as plt

col2ignore = ['Unnamed: 0', 'exp_name', 'id', 'base_name', 'date', 
              'original_video', 'directory', 'strain',
       'strain_description', 'allele', 'gene', 'chromosome',
       'tracker', 'sex', 'developmental_stage', 'ventral_side', 'food',
       'habituation', 'experimenter', 'arena', 'exit_flag', 'experiment_id',
       'n_valid_frames', 'n_missing_frames', 'n_segmented_skeletons',
       'n_filtered_skeletons', 'n_valid_skeletons', 'n_timestamps',
       'first_skel_frame', 'last_skel_frame', 'fps', 'total_time',
       'microns_per_pixel', 'mask_file_sizeMB', 'skel_file', 'frac_valid',
       'worm_index', 'n_frames', 'n_valid_skel', 'first_frame']


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
    cross_validation_fold = 5
    base_strains = ['JU393', 'ED3054', 'JU394', 
                     'N2', 'JU440', 'ED3021', 'ED3017', 
                     'JU438', 'JU298', 'JU345', 'RC301', 
                     'AQ2947', 'ED3049',
                     'LSJ1', 'JU258', 'MY16', 
                     'CB4852', 'CB4856', 'CB4853',
                     ]
    save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    feat_files = {'OW_old' : 'ow_features_old_SWDB.csv'}

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
        feat_data[db_name] = feats
        
    col2ignore_r = col2ignore + ['strain_base_id', 'strain_id', 'set_type']
    #%% 
    #SET TO TRUE TO FIT A MODEL USING ALL THE FEATURES
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
    #SET TO TRUE TO FIT TO SELECT FEATURES IN BASE OF THE CLASSIFICATION ACCURACY
    if False:
        #I am using less trees and trials too speed up things
        n_trials = 20 
        n_trees = 150
        
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
                
                
                all_scores = []
                for iif, ff in enumerate(col_feats):
                    if ff in selected_feats:
                        continue
                    
                    start = time.time()
                    c_feats = selected_feats + [ff]
                    
                    args = feats_r, c_feats, id_index_str, n_samples, cross_validation_fold, n_trees
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
    #calculate feature accuracy changes depending on the number of features
    selected_feats = ['midbody_crawling_amplitude_abs', 'foraging_amplitude_abs', 'hips_bend_mean_pos', 'midbody_width_forward', 'neck_bend_sd_forward_abs', 'midbody_speed_pos', 'foraging_speed_pos', 'eigen_projection_4_forward_neg', 'bend_count_backward', 'tail_crawling_frequency_pos', 'length_forward', 'tail_bend_mean_forward_neg', 'head_bend_sd_forward_pos', 'midbody_bend_sd', 'head_bend_sd_neg', 'tail_tip_motion_direction_backward_pos', 'eigen_projection_3_paused_abs', 'head_tip_speed_paused_pos', 'eigen_projection_1_forward_neg', 'eigen_projection_5_forward_pos', 'head_tip_speed_forward', 'tail_speed_forward_neg', 'tail_bend_mean_forward', 'upsilon_turns_time_pos', 'path_range_paused', 'midbody_crawling_frequency', 'head_motion_direction_forward', 'width_length_ratio_backward', 'head_motion_direction_pos', 'path_curvature_paused']
    
    feats = feat_data['OW_old']
    id_index_str = 'strain_base_id'
    
    n_batch= mp.cpu_count()
    p = mp.Pool(n_batch)
    
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    good = ~feats['strain_base_id'].isnull()
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
    
    
    yy = [x.mean() for x in all_scores]
    err = [x.std() for x in all_scores]
    
    plt.figure()
    plt.errorbar(np.arange(1, len(yy)+1), yy, yerr=err)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.savefig('classification_accuracy.png')
    
