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
import matplotlib.pylab as plt
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
    
    
    feat_data = {}
    
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
        
        
        feat_data[db_name] = feats
    #%% create a dataset with all the features
    feats = feat_data['OW']
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    feats = feats[col_feats + ['base_name']]
    feat_data['all'] = feat_data['tierpsy'].merge(feats, on='base_name')
    #%%
    for db_name, feats in feat_data.items():
        if db_name != 'all':
            continue
        
        #clustermap
        dat = feats.groupby('strain').agg('median')
        #dat = feats.copy()
        #dat.index = dat['base_name']
        
        col_val = [x for x in dat.columns if not x in col2ignore_r]
        dat = dat[col_val]
        
        for mm in ['ward', 'complete']:#['single', 'average', 'complete', 'centroid', 'median', 'ward']:
            sns.clustermap(dat, method=mm)
            plt.title(mm)
            plt.show()
            
        
        
    