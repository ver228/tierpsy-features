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

def _h_process_row(dd):
    irow, row = dd
    print(irow)
    bn = os.path.join(row['results_dir'], row['base_name'])
    fname_OW = bn + '_features.hdf5'
    fname_tierpsy = bn + '_featuresN.hdf5'
    
    with pd.HDFStore(fname_OW) as fid:
        OW_feats = fid['/features_summary/medians']
    
    with pd.HDFStore(fname_tierpsy) as fid:
        tierpsy_feats = fid['/timeseries_data']
        
        
    irow, OW_feats, tierpsy_feats


if __name__ == '__main__':
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
    ORDER BY frac_valid
    '''
    experiments_df = pd.read_sql(sql, con=conn)
    
    
    n_batch= mp.cpu_count()
    p = mp.Pool(n_batch)
    all_stats = list(p.map(_h_process_row, experiments_df.iterrows()))
    