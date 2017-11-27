#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:55:02 2017

@author: ajaver
"""
import pandas as pd
import glob
import os
from tierpsy_features import ventral_signed_columns

import sys

sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
from misc import get_rig_experiments_df


def read_tierpsy_feats(experiments_df):
    all_stats = []
    for irow, row in experiments_df.iterrows():
        print(irow+1, len(experiments_df))
        features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    
        with pd.HDFStore(features_file, 'r') as fid:
            features_stats = fid['/features_stats']
        features_stats['experiment_id'] = row['id']
    
        all_stats.append(features_stats)
    all_stats = pd.concat(all_stats)
    
    feat_df = all_stats.pivot(index='experiment_id', columns='name', values='value')
    
    #remove ventral signed features
    valid_feats = [x for x in feat_df.columns if not any(x.startswith(f) and not 'abs' in x for f in ventral_signed_columns)]
    feat_df = feat_df[valid_feats]
    return feat_df

if __name__ == '__main__':
    save_dir = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/results/dementia_nanoparticles'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Nell/'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
    
    
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    set_type = 'featuresN'
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**', '*' + f_ext), recursive = True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments_df = get_rig_experiments_df(features_files, csv_files)
    experiments_df = experiments_df.sort_values(by='video_timestamp').reset_index()  
    experiments_df['id'] = experiments_df.index
    experiments_df = experiments_df[['id', 'strain', 'n_worms', 'directory', 'base_name', 'exp_name']]
    
    tierpsy_feats = read_tierpsy_feats(experiments_df)
    dd = experiments_df.join(tierpsy_feats)
    save_name = os.path.join(save_dir, 'nell_tierpsy_features.csv')
    dd.to_csv(save_name)
    
