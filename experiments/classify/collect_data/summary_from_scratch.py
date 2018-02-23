#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:01:01 2017

@author: ajaver
"""
import sys
import glob

import pymysql
import pandas as pd
import os
import multiprocessing as mp

from helper_collect_tierspy import process_feat_tierpsy_file
from helper_collect_ow import process_ow_file

def _tierpsy_process_row(data_in):
    irow, row = data_in
    fname = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    
    print(irow+1, os.path.basename(fname))
    try:
        feat_stats = process_feat_tierpsy_file(fname)
        feat_stats['experiment_id'] = int(row['id'])
        
        return feat_stats
    except:
        return None
    
    
def read_tierpsy_feats(experiments_df):
    
    n_batch= mp.cpu_count()
    
    p = mp.Pool(n_batch)
    all_stats = list(p.map(_tierpsy_process_row, experiments_df.iterrows()))
    #all_stats = list(map(_tierpsy_process_row, experiments_df.iterrows()))
    
    all_stats = [x for x in all_stats if x is not None]
    all_stats = pd.concat(all_stats, axis=1).T
    all_stats.index = all_stats['experiment_id']
    
    return all_stats

def _ow_process_row(data_in):
    irow, row = data_in
    fname = os.path.join(row['directory'], row['base_name'] + '_features.hdf5')
    
    print(irow+1, os.path.basename(fname))
    try:
        feat_stats = process_ow_file(fname)
        feat_stats['experiment_id'] = int(row['id'])
        
        return feat_stats
    except:
        return None
    
    
def read_ow_feats(experiments_df):
    
    n_batch= mp.cpu_count()
    
    p = mp.Pool(n_batch)
    all_stats = list(p.map(_ow_process_row, experiments_df.iterrows()))
    #all_stats = list(map(_tierpsy_process_row, experiments_df.iterrows()))
    
    all_stats = [x for x in all_stats if x is not None]
    all_stats = pd.concat(all_stats, axis=1).T
    all_stats.index = all_stats['experiment_id']
    
    return all_stats

#%%
def get_SWDB_feats():
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
    AND experimenter <> "Celine N. Martineau, Bora Baskaner"
    '''
    #ingnore Celine's data for this dataset
    
    experiments_df = pd.read_sql(sql, con=conn)
    experiments_df = experiments_df.rename(columns={'results_dir':'directory'})
    experiments_df.index = experiments_df['id']

    tierpsy_feats_f = read_tierpsy_feats(experiments_df)
    
    del tierpsy_feats_f['experiment_id']
    save_name = os.path.join(save_dir, 'tierpsy_features_full_SWDB.csv')
    dd = experiments_df.join(tierpsy_feats_f)
    dd.to_csv(save_name, index_label=False)
    
    ow_feats_f = read_ow_feats(experiments_df)
    
    del ow_feats_f['experiment_id']
    save_name = os.path.join(save_dir, 'ow_features_full_SWDB.csv')
    dd = experiments_df.join(ow_feats_f)
    dd.to_csv(save_name, index_label=False)


def ini_experiments_df():
    #sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
    d_path = os.path.join(os.environ['HOME'], 'Github/process-rig-data/process_files')
    sys.path.append(d_path)
    from misc import get_rig_experiments_df

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
#    get_SWDB_feats()

    save_dir = './'
    experiments_df = ini_experiments_df()
    experiments_df = experiments_df[['id', 'strain', 'directory', 'base_name', 'exp_name']]
    experiments_df.index = experiments_df['id']
#    
#    tierpsy_feats = read_tierpsy_feats(experiments_df)
#    dd = experiments_df.join(tierpsy_feats)
#    save_name = os.path.join(save_dir, 'tierpsy_features_full_CeNDR.csv')
#    dd.to_csv(save_name)
    
    ow_feats = read_ow_feats(experiments_df)
    dd = experiments_df.join(ow_feats)
    save_name = os.path.join(save_dir, 'ow_features_full_CeNDR.csv')
    dd.to_csv(save_name)