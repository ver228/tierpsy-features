#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:01:01 2017

@author: ajaver
"""

import glob
import sys
import os
import tables
import pandas as pd
import numpy as np
from tierpsy_features import ventral_signed_columns
from functools import partial
import multiprocessing as mp

sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
from misc import get_rig_experiments_df

#the sign of this features is related with the ventral orientation. This is not defined in the multiworm case.
_ow_signed_ventral_feats = ['head_bend_mean', 'neck_bend_mean', 
                'midbody_bend_mean', 'hips_bend_mean', 
                'tail_bend_mean', 'head_bend_sd', 'neck_bend_sd', 
                'midbody_bend_sd', 'hips_bend_sd', 'tail_bend_sd', 
                'tail_to_head_orientation', 'head_orientation', 
                'tail_orientation', 'eigen_projection_1', 
                'eigen_projection_2', 'eigen_projection_3', 
                'eigen_projection_4', 'eigen_projection_5', 
                'eigen_projection_6', 'head_tip_motion_direction', 
                'head_motion_direction', 'midbody_motion_direction', 
                'tail_motion_direction', 'tail_tip_motion_direction', 
                'foraging_amplitude', 'foraging_speed', 
                'head_crawling_amplitude', 'midbody_crawling_amplitude', 
                'tail_crawling_amplitude', 'head_crawling_frequency', 
                'midbody_crawling_frequency', 'tail_crawling_frequency', 
                'path_curvature']

_ow_bad_feats = ['bend_count', 'head_orientation', 'tail_to_head_orientation', 'tail_orientation']


def _ow_read_feat_events(features_file):
    
    features_events = {}
    with tables.File(features_file, 'r') as fid:
        node = fid.get_node('/features_events')
        for worn_n in node._v_children.keys():
            worm_node = fid.get_node('/features_events/' + worn_n)
            
            for feat in worm_node._v_children.keys():
                if not feat in features_events:
                    features_events[feat] = {}
                dat = fid.get_node(worm_node._v_pathname, feat)[:]
                features_events[feat][worn_n] = dat
                
    features_events = {k:np.concatenate(list(d.values()))for k,d in features_events.items()}
    
    return features_events

def _ow_get_feat_stats(features_timeseries, features_events, FRAC_MIN):
    #%%
    #columns that correspond to indexes (not really features)
    index_cols =['worm_index','timestamp','skeleton_id','motion_modes']
    valid_feats = [x for x in features_timeseries.columns if not x in index_cols]
   
    q_vals = [10, 50, 90]
    iqr_limits = [25, 75]
    valid_q = q_vals + iqr_limits
    
    def _get_stats(feat, dat):
        if np.sum(~np.isnan(dat))/dat.size > FRAC_MIN:
            q_r = np.nanpercentile(dat, valid_q)
            
            rr = []
            for ii, q in enumerate(q_vals):
                rr.append((q_r[ii], '{}_{}th'.format(feat, q)))
            rr.append((q_r[-1] - q_r[-2], feat + '_IQR'))
            return rr
    
    
    r_stats_l = [_get_stats(feat, features_timeseries[feat]) for feat in valid_feats]
    
    r_stats_e = [_get_stats(feat, dat) for feat, dat in features_events.items()]

    
    #the motion_modes is a bit different. It is -1 if the worm is going backwards,
    # 0 if it is paused and 1 if it is going forward
    motion_d = {-1:'backward', 1:'forward', 0:'paused'}
    motion_modes = features_timeseries['motion_modes']
    nn = motion_modes.count()
    
    r_stats_m = []
    if nn/motion_modes.size > FRAC_MIN:
        for k, v in motion_d.items():
            val = np.sum(motion_modes==k)/nn
            feat = 'motion_mode_{}_fraction'.format(v)
            r_stats_m.append((val, feat))
    
    
    r_stats = r_stats_e + r_stats_l
    r_stats = [x for x in r_stats if x is not None]
    r_stats = sum(r_stats, [])
    r_stats += r_stats_m

    r_stats = pd.DataFrame(r_stats, columns=['value', 'name'])
    return r_stats

def _h_ow_process_row(dd, FRAC_MIN):
    irow, row = dd
    print(irow+1)
    features_file = os.path.join(row['directory'], row['base_name'] + '_features.hdf5')
    
    with pd.HDFStore(features_file, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
        
    features_events = _ow_read_feat_events(features_file)
    features_stats = _ow_get_feat_stats(features_timeseries, features_events, FRAC_MIN=FRAC_MIN)
    features_stats['experiment_id'] = row['id']
    
    return features_stats

def read_ow_feats(experiments_df, FRAC_MIN=0.8):
    n_batch= mp.cpu_count()
    p = mp.Pool(n_batch)
    row_fun = partial(_h_ow_process_row, FRAC_MIN = FRAC_MIN)
    all_stats = list(p.map(row_fun, experiments_df.iterrows()))
    
    if False:
        all_stats = []
        for dd in experiments_df.iterrows():
            features_stats = _h_ow_process_row(dd)
            all_stats.append(features_stats)
    
    all_stats = pd.concat(all_stats)
    feat_df = all_stats.pivot(index='experiment_id', columns='name', values='value')

    return feat_df

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

def ini_experiments_df():
    exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')

    set_type = 'featuresN'
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments_df = get_rig_experiments_df(features_files, csv_files)
    experiments_df = experiments_df.sort_values(by='video_timestamp').reset_index()  
    experiments_df['id'] = experiments_df.index
    return experiments_df

if __name__ == '__main__':
    save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/CeNDR'
    
    experiments_df = ini_experiments_df()
    experiments_df = experiments_df[['id', 'strain', 'directory', 'base_name', 'exp_name']]
    experiments_df.index = experiments_df['id']
    
    tierpsy_feats = read_tierpsy_feats(experiments_df)
    dd = experiments_df.join(tierpsy_feats)
    save_name = os.path.join(save_dir, 'tierpsy_features_CeNDR.csv')
    dd.to_csv(save_name)
    
    ow_feats = read_ow_feats(experiments_df)
    dd = experiments_df.join(ow_feats)
    save_name = os.path.join(save_dir, 'ow_features_CeNDR.csv')
    dd.to_csv(save_name)
    