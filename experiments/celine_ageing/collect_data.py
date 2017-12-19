#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:11:40 2017

@author: ajaver
"""
import os
import glob
import re
import pandas as pd

def get_file_parts(fnames):
    all_data = []
    for feat_file in fnames:
        fdir, bn = os.path.split(feat_file)
        bn = bn.replace('_featuresN.hdf5', '')
        parts = os.path.basename(bn).split('_')
        
        Rn = int(parts[1].split('#')[-1])
        strain, worm_id = parts[2].split('#')
        worm_id = int(worm_id)
        
        if 'L4' in parts[3].upper():
            day = 0
        else:
            day = int(re.sub("[^0-9]", "", parts[3]))
        v_orientation = parts[4]
        
        ts = pd.Timestamp(*[int(x) for x in parts[5:12] if x])
        
        dd = (fdir, bn, Rn, strain, worm_id, day, v_orientation, ts)
        all_data.append(dd)
        
    df = pd.DataFrame(all_data, columns=['directory', 'base_name', 'replicated_n', 'strain', 'worm_id', 'day', 'ventral_orientation', 'timestamp'])
    df = df.sort_values(['strain', 'replicated_n', 'worm_id', 'day']).reset_index(drop=True)
    
    df['id'] = df.index
    return df

def read_tierpsy_feats(experiments_df):
    all_stats = []
    for irow, row in experiments_df.iterrows():
        print(irow+1, len(experiments_df))
        features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    
        with pd.HDFStore(features_file, 'r') as fid:
            if not '/features_stats' in fid:
                continue
            
            features_stats = fid['/features_stats']
        features_stats['experiment_id'] = row['id']
    
        all_stats.append(features_stats)
    all_stats = pd.concat(all_stats)
    
    feat_df = all_stats.pivot(index='experiment_id', columns='name', values='value')
    
    #remove ventral signed features
    #valid_feats = [x for x in feat_df.columns if not any(x.startswith(f) and not 'abs' in x for f in ventral_signed_columns)]
    #feat_df = feat_df[valid_feats]
    return feat_df

if __name__ == '__main__':
    main_dir = '/Volumes/behavgenom_archive$/Celine/results'
    
    fnames = glob.glob(os.path.join(main_dir, '**', '*_featuresN.hdf5'), recursive=True)
    experiments_df = get_file_parts(fnames)
    
    experiments_df.to_csv('ageing_celine.csv', index=False)
    
    feat_df = read_tierpsy_feats(experiments_df)
    feat_df.to_csv('ageing_celine_feats.csv')
    