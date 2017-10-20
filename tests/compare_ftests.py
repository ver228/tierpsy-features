#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:20:36 2017

@author: ajaver
"""
import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

save_dir = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/comparisons/CeNDR/'

min_n_videos = 3

feat_files = {
        'OW' : 'ow_features_CeNDR.csv',
        'tierpsy' :'tierpsy_features_CeNDR.csv'
        }

col2ignore = ['Unnamed: 0', 'id', 'directory', 'base_name', 'exp_name']

all_features = {}
for db_name, feat_file in feat_files.items():
    print(db_name)
    feats = pd.read_csv(save_dir + feat_file)
    
    
    dd = feats.isnull().mean()
    col2remove =  dd.index[(dd>0.25).values].tolist()
    feats = feats[[x for x in feats if x not in col2remove]]
    all_features[db_name] = feats

assert (all_features['OW']['base_name'] == all_features['tierpsy']['base_name']).all()

#%%
dd = all_features['OW']['strain'].value_counts()
good_strains = dd.index[(dd>=min_n_videos).values].values

for db_name, feats in all_features.items():
    
    feats = feats[feats['strain'].isin(good_strains)]
    feats_g = [(s,dat) for s,dat in feats.groupby('strain')]
    
    all_data = []
    for ii, (s, s_dat) in enumerate(feats_g):
        print(ii, s)
        s_dat = s_dat.fillna(s_dat.median())
        if np.any(s_dat.isnull()):
            print('bad')
            continue
        
        all_data.append(s_dat)
    feats = pd.concat(all_data).sort_index()
    all_features[db_name] = feats
#%%
valid_ind = None
for db_name, feats in all_features.items():
    if valid_ind is None:
        valid_ind = set(feats.index)
    else:
        valid_ind = valid_ind & set(feats.index)

for db_name, feats in all_features.items():
    all_features[db_name] = feats.loc[valid_ind]
#%%
fstats_comparsions = {}
for db_name, feats in all_features.items():
    
    feats_g = feats.groupby('strain')
    
    all_fstats = []
    for ii, col in enumerate(feats):
        if col in col2ignore + ['strain']:
            continue
        print(ii, col)
        dat = [s_dat[col].values for s, s_dat in feats_g]
        dat = [x for x in dat if ~np.all(np.isnan(x))]
        
        
        fstats, pvalue = f_oneway(*dat)
        
        all_fstats.append((col, fstats, pvalue))
    
    fstats_df = pd.DataFrame(all_fstats, columns = ['features', 'fstat', 'pvalue'])
    fstats_df = fstats_df.sort_values(by='fstat')
    
    fstats_comparsions[db_name] = fstats_df
#%%
plt.figure()
for db_name, dat in fstats_comparsions.items():
    dat = fstats_comparsions[db_name]
    yy = dat['fstat'].values
    plt.plot(yy, label=db_name)
plt.legend(loc=2)
#%%
plt.figure()
for db_name, dat in fstats_comparsions.items():
    yy = np.log10(dat['pvalue'].values)
    plt.plot(yy, label=db_name)
plt.legend(loc=1)

#%%
for db_name, feats in all_features.items():
    bn = feat_files[db_name]
    fname = os.path.join(save_dir, 'F_' + bn)
    feats.to_csv(fname)

