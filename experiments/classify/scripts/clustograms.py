#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:58:04 2018

@author: ajaver
"""


import os
import numpy as np
import pandas as pd
import seaborn as sns
from compare_ftests import col2ignore

if __name__ == '__main__':
    
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../data/SWDB'
    feat_files = {
            'OW' : 'F0.025_ow_features_old_SWDB.csv',
            'tierpsy' :'F0.025_tierpsy_features_SWDB.csv'
            }
    #%%
    col2ignore_r = col2ignore + ['strain_id', 'set_type']
    
    for db_name, bn in feat_files.items():
        print(db_name)
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        
        col_val = [x for x in feats.columns if x not in col2ignore_r]
        
        dd = feats[col_val]
        z = (dd-dd.mean())/(dd.std())
        feats[col_val] = z
        
        #clustermap
        dat = feats.groupby('strain').agg('median')
        col_val = [x for x in dat.columns if not x in col2ignore_r]
        dat = dat[col_val]
        sns.clustermap(dat)