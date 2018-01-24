#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:01:01 2017

@author: ajaver
"""

import pymysql
import pandas as pd
import os
import multiprocessing as mp

from functools import partial
from summary_CeNDR import read_ow_feats

from tierpsy_features.summary_stats import _test_get_feat_stats_all
from tierpsy.helper.params import read_fps

def _h_process_row(dd):
    irow, row = dd
    print(irow)
    bn = os.path.join(row['directory'], row['base_name'])
    fname_OW = bn + '_features.hdf5'
    fname_tierpsy = bn + '_featuresN.hdf5'
    
    with pd.HDFStore(fname_OW) as fid:
        OW_feats = fid['/features_summary/medians']
    
    with pd.HDFStore(fname_tierpsy) as fid:
        tierpsy_feats = fid['/features_stats']
        tierpsy_feats['index'] = irow
        
    return irow, OW_feats, tierpsy_feats

def _tierpsy_process_row(data_in):
    irow, row = data_in
    print(irow+1)
    fname = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    with pd.HDFStore(fname, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features']
    fps = read_fps(fname)
    
    feat_stats = _test_get_feat_stats_all(timeseries_data, blob_features, fps)
    
    feat_stats['experiment_id'] = int(row['id'])
    
    return feat_stats
    
def read_tierpsy_feats(experiments_df):
    
    n_batch= mp.cpu_count()
    
    p = mp.Pool(n_batch)
    all_stats = list(p.map(_tierpsy_process_row, experiments_df.iterrows()))
    
    all_stats = pd.concat(all_stats, axis=1).T
    all_stats.index = all_stats['experiment_id']
    
    return all_stats
#%%
if __name__ == '__main__':
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB'
    save_dir = './'
    
    conn = pymysql.connect(host='localhost', database='single_worm_db')

    sql = '''
    SELECT *, 
    CONCAT(results_dir, '/', base_name, '_skeletons.hdf5') AS skel_file,
    n_valid_skeletons/(total_time*fps) AS frac_valid
    FROM experiments_full AS e
    JOIN results_summary ON e.id = experiment_id
    WHERE total_time < 905
    AND total_time > 295
    AND strain != '-N/A-'
    AND exit_flag = 'END'
    AND n_valid_skeletons > 120*fps
    '''
    experiments_df = pd.read_sql(sql, con=conn)
    experiments_df = experiments_df.rename(columns={'results_dir':'directory'})
    experiments_df.index = experiments_df['id']
    
    tierpsy_feats_f = read_tierpsy_feats(experiments_df)
    del tierpsy_feats_f['experiment_id']
    save_name = os.path.join(save_dir, 'tierpsy_features_full_SWDB.csv')
    dd = experiments_df.join(tierpsy_feats_f)
    dd.to_csv(save_name, index_label=False)
#%%
#    n_batch= mp.cpu_count()
#    p = mp.Pool(n_batch)
#    all_stats = list(p.map(_h_process_row, experiments_df.iterrows()))
#    
#    n_exp, ow_feats_old, tierpsy_feats = zip(*all_stats)
#    ow_feats_old = pd.concat(ow_feats_old, ignore_index=True)
#    ow_feats_old.index = n_exp
#    save_name = os.path.join(save_dir, 'ow_features_old_SWDB.csv')
#    dd = experiments_df.join(ow_feats_old)
#    dd.to_csv(save_name, index_label=False)
#    
#    tierpsy_feats = pd.concat(tierpsy_feats, ignore_index=True)
#    tierpsy_feats = tierpsy_feats.pivot(index='index', columns='name', values='value')
#    save_name = os.path.join(save_dir, 'tierpsy_features_SWDB.csv')
#    dd = experiments_df.join(tierpsy_feats)
#    dd.to_csv(save_name, index_label=False)
#    
#    ow_feats = read_ow_feats(experiments_df, FRAC_MIN=0.25)
#    save_name = os.path.join(save_dir, 'ow_features_SWDB.csv')
#    dd = experiments_df.join(ow_feats)
#    dd.to_csv(save_name)
    