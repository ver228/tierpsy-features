#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:23:00 2017

@author: ajaver
"""
import os
import pandas as pd
import numpy as np
import tables

CeNDR_base_strains = ['N2', 'ED3017', 'CX11314', 'LKC34', 'MY16', 'DL238', 'JT11398', 'JU775',
       'JU258', 'MY23', 'EG4725', 'CB4856']


save_dir = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/comparisons/CeNDR/'

feat_files = {
        'OW' : 'ow_features_CeNDR.csv',
        'tierpsy' :'tierpsy_features_CeNDR.csv'
        }

#%%
main_file = '/Users/ajaver/Desktop/CeNDR_skel_smoothed.hdf5'

with pd.HDFStore(main_file, 'r') as fid:
    df1 = fid['/skeletons_groups']
    df3 = fid['/experiments_data']
    strain_codes = fid['/strains_codes']

cols_to_use = df3.columns.difference(df1.columns)
df3 = df3[cols_to_use]
df3 = df3.rename(columns={'id' : 'experiment_id'})
df = df1.join(df3.set_index('experiment_id'), on='experiment_id')
#%%
valid_indices = {}
with tables.File(main_file, 'r') as fid:
    for set_type in ['train', 'test', 'val']:
        dd = '/index_groups/' + set_type
        if dd in fid:
            v_ind = fid.get_node(dd)[:]
            #use previously calculated indexes to divide data in training, validation and test sets
            valid_indices[set_type] = df.loc[v_ind, 'base_name'].unique()
            
assert not (set(valid_indices['test']) & set(valid_indices['train']))
#%%
sets2test = {}
for db_name, bn in feat_files.items():
    fname = os.path.join(save_dir, 'F_' + bn)
    feats = pd.read_csv(fname)
    
    #s_dict = {s:ii for _, (s,ii) in strain_codes.iterrows()}
    #
    ss = np.sort(feats['strain'].unique())
    s_dict = {s:ii for ii,s in enumerate(ss)}
    feats['strain_id'] = feats['strain'].map(s_dict)
    
    sets2test[db_name] = {}
    for set_type, v_ind in valid_indices.items():
        sets2test[db_name][set_type] = feats[feats['base_name'].isin(v_ind)]
#%%
from sklearn.ensemble import RandomForestClassifier

classifiers_d = {}
for db_name, datasets  in sets2test.items():
    print(db_name, datasets['train'].shape)
    
    col2ignore = ['Unnamed: 0', 'id', 'directory', 'base_name', 'exp_name', 'strain']
    col_feats = [x for x in datasets['train'].columns if x not in col2ignore]
    
    X_train = datasets['train'][col_feats].values
    Y_train = datasets['train']['strain_id'].values
    
    x_test = datasets['test'][col_feats].values
    y_test = datasets['test']['strain_id'].values
    
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, Y_train)
    
    proba = clf.predict_proba(x_test)
    top_pred = np.argsort(proba, axis=1)[: ,::-1]
    preds = top_pred==y_test[:, np.newaxis]
    print(db_name, preds[:,0].mean())
    
    
    classifiers_d[db_name] = clf
    
#%%
CeNDR_base_strains_id = {x:i for i,x in enumerate(CeNDR_base_strains)}

classifiers_d = {}
for db_name, datasets  in sets2test.items():
    
    
    col2ignore = ['Unnamed: 0', 'id', 'directory', 'base_name', 'exp_name', 'strain']
    col_feats = [x for x in datasets['train'].columns if x not in col2ignore]
    
    ss = datasets['train']
    ss = ss[ss['strain'].isin(CeNDR_base_strains)]
    X_train = ss[col_feats].values
    Y_train = ss['strain'].map(CeNDR_base_strains_id).values
    print(db_name, ss.shape)
    
    
    ss = datasets['test']
    ss = ss[ss['strain'].isin(CeNDR_base_strains)]
    x_test = ss[col_feats].values
    y_test = ss['strain'].map(CeNDR_base_strains_id).values
    print(db_name, ss.shape)
    
    clf = RandomForestClassifier(n_estimators=10000)
    clf.fit(X_train, Y_train)

    proba = clf.predict_proba(x_test)
    top_pred = np.argsort(proba, axis=1)[: ,::-1]
    preds = top_pred==y_test[:, np.newaxis]
    print(db_name, preds[:,0].mean())
    
    classifiers_d[db_name] = clf
    