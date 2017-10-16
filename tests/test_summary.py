#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:01:01 2017

@author: ajaver
"""

import glob
import sys
import os
import pandas as pd

from tierpsy_features.summary_stats import get_feat_stats, get_df_quantiles, _filter_ventral_features
from tierpsy_features import timeseries_feats_columns

if __name__ == '__main__':
    
    from tierpsy.helper.params import read_ventral_side
    
    sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
    from misc import get_rig_experiments_df
    exp_set_dir = '/Users/ajaver/OneDrive - Imperial College London/swiss_strains'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
    
    set_type = 'featuresN'
    
    save_dir = './results_{}'.format(set_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments = get_rig_experiments_df(features_files, csv_files)
    
    info_cols = ['strain']
    
    fnames = [os.path.join(row['directory'], row['base_name'] + f_ext)
                for _, row in experiments.iterrows()]
    info_df = experiments[info_cols]
    
    assert info_df.shape[0] == len(fnames)
    
    all_data = []
    
    
    for fname, (ii, row) in zip(fnames, info_df.iterrows()):
        print(ii+1, len(info_df))
        
        ventral_side = read_ventral_side(fname)
        
        with pd.HDFStore(fname, 'r') as fid:
            fps = fid.get_storer('/trajectories_data').attrs['fps']
            timeseries_data = fid['/timeseries_data']
            blob_features = fid['/blob_features']
        
        feat_stats = get_feat_stats(timeseries_data, fps, is_normalize = False)
        
        #this is a dirty solution to avoid duplicates but we are testing right now
        feat_stats_n = get_feat_stats(timeseries_data, fps, is_normalize = True)
        feat_stats_n = feat_stats_n[[x for x in feat_stats_n.index if x not in feat_stats.index]]
        
        # another dirty solution to add features with ventral sign
        ventral_feats = _filter_ventral_features(timeseries_feats_columns)
        if ventral_side == 'clockwise':
            timeseries_data[ventral_feats] *= -1
        
        feat_stats_v = get_df_quantiles(timeseries_data, 
                         feats2check = ventral_feats, 
                         is_abs_ventral = False, 
                         is_normalize = False)
        
        
        feat_stats_v_n = get_df_quantiles(timeseries_data, 
                         feats2check = ventral_feats, 
                         is_abs_ventral = False, 
                         is_normalize = True)
        feat_stats_v_n = feat_stats_v_n[[x for x in feat_stats_v_n.index if x not in feat_stats_v.index]]
        
        blob_feats = [
               'area', 'perimeter', 'box_length', 'box_width',
               'quirkiness', 'compactness', 'solidity',
               'hu0', 'hu1', 'hu2', 'hu3', 'hu4',
               'hu5', 'hu6'
               ]
        blob_stats = get_df_quantiles(blob_features, feats2check = blob_feats)
        blob_stats.index = ['blob_' + x for x in blob_stats.index]
        
        exp_feats = pd.concat((feat_stats, feat_stats_n, blob_stats, feat_stats_v, feat_stats_v_n))
        
        info_zip = zip(*[row.tolist()]*len(exp_feats))
        all_data += list(zip(*info_zip, exp_feats, exp_feats.index))
        
        print(len(all_data))
        
        break