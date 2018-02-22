#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:33:20 2018

@author: ajaver
"""
import os
import pandas as pd
import numpy as np


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
#%%
def read_feats():
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../../data/SWDB'
    feat_files = {
            'tierpsy' : 'F0.025_tierpsy_features_full_SWDB.csv',
            'OW' : 'F0.025_ow_features_full_SWDB.csv',
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
    
    # create a dataset with all the features
    feats = feat_data['OW']
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    feats = feats[col_feats + ['base_name']]
    feats.columns = [x if x == 'base_name' else 'ow_' + x for x in feats.columns]
    feat_data['all'] = feat_data['tierpsy'].merge(feats, on='base_name')
    
    # scale data
    for db_name, feats in feat_data.items(): 
        col_val = [x for x in feats.columns if x not in col2ignore_r]
        dd = feats[col_val]
        z = (dd-dd.mean())/(dd.std())
        feats[col_val] = z
        feat_data[db_name] = feats
    
    return feat_data, col2ignore_r

def get_core_features(feat_data, col2ignore_r):
    assert ('OW' in feat_data) and ('tierpsy' in feat_data) 
    
    
    def _remove_end(col_v, p2rev):
        col_v_f = []
        for x in col_v:
            xf = x
            for p in p2rev:
                if x.endswith(p):
                    xf = xf[:-len(p)]
                    break
            col_v_f.append(xf)
        
        return list(set(col_v_f))
    
    # obtain the core features from the feature list
    core_feats = {}
    
    #OW
    col_v = [x for x in feat_data['OW'].columns if x not in col2ignore_r]
    col_v = _remove_end(col_v, ['_abs', '_neg', '_pos'])
    col_v = _remove_end(col_v, ['_paused', '_forward', '_backward'])
    col_v = _remove_end(col_v, ['_distance', '_distance_ratio', '_frequency', '_time', '_time_ratio'])
    core_feats['OW'] = sorted(col_v)
    
    #tierpsy
    col_v = [x for x in feat_data['tierpsy'].columns if x not in col2ignore_r]
    col_v = list(set([x[2:] if x.startswith('d_') else x for x in col_v]))
    col_v = _remove_end(col_v, ['_10th', '_50th', '_90th', '_95th', '_IQR'])
    col_v = _remove_end(col_v, ['_w_forward', '_w_backward']) #where is paused??
    col_v = _remove_end(col_v, ['_abs'])
    col_v = _remove_end(col_v, ['_norm'])
    col_v = _remove_end(col_v, ['_frequency', '_fraction', '_duration', ])
    core_feats['tierpsy'] = sorted(col_v)
    
    #all
    core_feats['all']  = core_feats['tierpsy'] + ['ow_' + x for x in core_feats['OW']]
    return core_feats

def get_feat_group_indexes(core_feats_v, col_feats):
    '''
    Get the keys, i am assuming there is a core_feats for each of the col_feats
    '''
    
    c_feats_dict = {x:ii for ii, x in enumerate(core_feats_v)}
    
    #sort features by length. In this way I give priority to the longer feat e.g area_length vs area 
    core_feats_v = sorted(core_feats_v, key = len)[::-1]
    
    def _search_feat(feat):
        for core_f in core_feats_v:
            if feat.startswith(core_f):
                return c_feats_dict[core_f]
        #the correct index was not found return -1
        return -1 
        
    
    col_feats = [x[2:] if x.startswith('d_') else x for x in col_feats]
    f_groups_inds = np.array([_search_feat(x) for x in col_feats])
    
    
    return f_groups_inds, c_feats_dict