#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:05:10 2017

@author: ajaver
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from tierpsy_features.summary_stats import get_time_groups


if __name__ == '__main__':
    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    all_fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive = True)
    
    set_prefix = '4h'
    exp_valid = [7,8]
    time_ranges_m = [(n*15, (n+1)*15) for n in range(8)]
    
    print('****** {} ******'.format(set_prefix))
    save_dir_root = '/Users/ajaver/OneDrive - Imperial College London/development/test_041017b/Set_{}'
    fnames = [x for x in all_fnames if int(x.split('Experiment')[-1].split('/')[0]) in exp_valid]
            
    
    dat = []
    for fname in fnames:
        exp_key = fname.split(os.sep)[-2].split('_')[-1]
        exp_n = int(exp_key.split('co')[0][3:])
        cohort_n = int(exp_key.split('co')[-1])
        dat.append((exp_n, cohort_n))
    info_df = pd.DataFrame(dat, columns = ['exp_n', 'cohort_n'])
    
    
    bins_edges = np.arange(-3375, 6875, 750)#np.linspace(-3000, 6500, 21)
    #%%
    all_h = []
    for (iexp, info_row), fname in zip(info_df.iterrows(), fnames):
        print(iexp + 1, len(fnames))
        
        with pd.HDFStore(fname, 'r') as fid:
            fps = fid.get_storer('/trajectories_data').attrs['fps']
            df = fid['/timeseries_features']
        #%%
        df['time_group'] = get_time_groups(df['timestamp'], time_ranges_m, fps)
        df_g = df.groupby('time_group')
        for t, dat in df_g:
            if t < 0:
                continue
            h, _ = np.histogram(dat['dist_from_food_edge'].dropna(), bins_edges)
            
            d = info_row.copy()
            d['time_group'] = t
            all_h.append((d, h))
        #%%
        
        #%%
        
    #%%
    e_info, data = zip(*all_h)
    e_info = pd.concat(e_info, axis=1, ignore_index=True).T
    data = np.vstack(data)
    
    all_hist = []
    bins = bins_edges[:-1] + (bins_edges[1:]-bins_edges[:-1])/2
    for ii, (tt, t_g) in enumerate(e_info.groupby('time_group')):
        for n_cohort, e_g in t_g.groupby('cohort_n'):
            h = np.sum(data[e_g.index],axis=0)
            #h = h/np.sum(h)
            
            df = pd.DataFrame(list(zip(bins, h, h.size*[tt], h.size*[n_cohort])), 
                              columns = ['bins', 'counts', 'time_group', 'cohort_n'])
            
            all_hist.append(df)
    all_hist = pd.concat(all_hist, ignore_index=True)
    
    all_hist['bins'] /= 1e3
    sns.factorplot(
            x = 'bins',
            y = 'counts',
            hue = 'cohort_n',
            col = 'time_group',
            data = all_hist,
            col_wrap = 4,
            kind = 'point'
            )
#%%

    