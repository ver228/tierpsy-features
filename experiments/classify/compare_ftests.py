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
from functools import partial
import multiprocessing as mp

#col2ignore = ['Unnamed: 0', 'id', 'directory', 'base_name', 'exp_name']
col2ignore = ['Unnamed: 0', 'exp_name', 'id', 'base_name', 'date', 
              'original_video', 'directory', 'strain',
       'strain_description', 'allele', 'gene', 'chromosome',
       'tracker', 'sex', 'developmental_stage', 'ventral_side', 'food',
       'habituation', 'experimenter', 'arena', 'exit_flag', 'experiment_id',
       'n_valid_frames', 'n_missing_frames', 'n_segmented_skeletons',
       'n_filtered_skeletons', 'n_valid_skeletons', 'n_timestamps',
       'first_skel_frame', 'last_skel_frame', 'fps', 'total_time',
       'microns_per_pixel', 'mask_file_sizeMB', 'skel_file', 'frac_valid',
       'worm_index', 'n_frames', 'n_valid_skel', 'first_frame']




def _get_args(set_type):
    if set_type == 'CeNDR':
        MIN_N_VIDEOS = 3
        save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/CeNDR/'
        feat_files = {
                'OW' : 'ow_features_CeNDR.csv',
                'tierpsy' :'tierpsy_features_CeNDR.csv'
                }
    elif set_type == 'SWDB':
        MIN_N_VIDEOS = 10
        save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
        feat_files = {
                'OW_old' : 'ow_features_old_SWDB.csv',
                'OW' : 'ow_features_SWDB.csv',
                'tierpsy' :'tierpsy_features_SWDB.csv'
                }
    return MIN_N_VIDEOS, save_dir, feat_files

def _h_ftest(columns, feats_g):
    all_f = []
    print(len(columns))
    for col in columns:
        
        dat = [s_dat[col].values for s, s_dat in feats_g]
        dat = [x for x in dat if ~np.all(np.isnan(x))]
        
        
        fstats, pvalue = f_oneway(*dat)
        if ~np.isnan(pvalue):
            all_f.append((col, fstats, pvalue))
    return all_f

if __name__ == '__main__':
    MAX_FRAC_NAN = 0.25
    
    
    MIN_N_VIDEOS, save_dir, feat_files = _get_args('CeNDR')
    

    
    all_features = {}
    for db_name, feat_file in feat_files.items():
        print(db_name)
        feats = pd.read_csv(save_dir + feat_file)
        
        
        dd = feats.isnull().mean()
        col2remove =  dd.index[(dd>MAX_FRAC_NAN).values].tolist()
        feats = feats[[x for x in feats if x not in col2remove]]
        all_features[db_name] = feats
    
    assert (all_features['OW']['base_name'].values == all_features['tierpsy']['base_name'].values).all()
    
    #%%
    dd = all_features['OW']['strain'].value_counts()
    good_strains = dd.index[(dd>=MIN_N_VIDEOS).values].values
    
    for db_name, feats in all_features.items():
        feats = feats[feats['strain'].isin(good_strains)]
        #Imputate missing values. I am using the global median to be more conservative
        all_features[db_name] = feats.fillna(feats.median())
    
    #%%
    #    feats = feats[feats['strain'].isin(good_strains)]
    #    feats_g = [(s,dat) for s,dat in feats.groupby('strain')]
    #    
    #    all_data = []
    #    for ii, (s, s_dat) in enumerate(feats_g):
    #        print(ii, s)
    #        s_dat = s_dat.fillna(s_dat.median())
    #        if np.any(s_dat.isnull()):
    #            print('bad')
    #            continue
    #        
    #        all_data.append(s_dat)
    #    feats = pd.concat(all_data).sort_index()
    #    all_features[db_name] = feats
    #%%
    #select files that are present in all the features sets
    valid_ind = None
    for db_name, feats in all_features.items():
        if valid_ind is None:
            valid_ind = set(feats['base_name'].values)
        else:
            valid_ind = valid_ind & set(feats['base_name'].values)
    
    for db_name, feats in all_features.items():
        all_features[db_name] = feats[feats['base_name'].isin(valid_ind)]
    #%%
    n_batch= mp.cpu_count()
    p = mp.Pool(n_batch)
    #%%
    
    #all_stats = list(p.map(row_fun, experiments_df.iterrows()))
    
    
    
    fstats_comparsions = {}
    for db_name, feats in all_features.items():
        print(db_name)
        
        feats_g = feats.groupby('strain')
        valid_cols = [x for x in feats if x not in col2ignore]
        
        n_chuck = 50
        valid_cols = [valid_cols[x:x+n_chuck] for x in range(0, len(valid_cols), n_chuck)]
        
        func = partial(_h_ftest, feats_g=feats_g)
        all_fstats = list(p.map(func, valid_cols))
        #all_fstats = [x for x in all_fstats if x is not None]
        all_fstats = sum(all_fstats, [])
        fstats_df = pd.DataFrame(all_fstats, columns = ['features', 'fstat', 'pvalue'])
        fstats_df = fstats_df.sort_values(by='fstat')
        
        fstats_comparsions[db_name] = fstats_df
    #%%
    plt.figure()
    for db_name, dat in fstats_comparsions.items():
        print(db_name, dat.shape)
        dat = fstats_comparsions[db_name]
        yy = dat['fstat'].values
        plt.plot(yy, label=db_name)
    plt.legend(loc=2)
    #%%
    plt.figure()
    for db_name, dat in fstats_comparsions.items():
        print(db_name, dat.shape)
    
        yy = dat['pvalue'].values
        eps = yy[yy!=0].min()
        yy = np.clip(yy, eps, 1)
        yy = np.log10(yy)
        plt.plot(yy, label = db_name)
    plt.legend(loc=1)
    
    #%%
    for db_name, feats in all_features.items():
        bn = feat_files[db_name]
        fname = os.path.join(save_dir, 'F_' + bn)
        feats.to_csv(fname)
    
    #%%
    
    
    