#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:55:02 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import glob
import os

from tierpsy.helper.params import read_fps

import matplotlib.pylab as plt
import matplotlib
matplotlib.style.use('ggplot')

#%%
def averages_by_time(df, fps):

    index_columns = ['worm_index', 'timestamp', 'timestamp_m']
    feat_columns = [x for x in df.columns if not x in index_columns]
    df
    def top10(x):
        return np.nanpercentile(x, 10)
    def top90(x):
        return np.nanpercentile(x, 90)
    
    def mad(x):
        #median absolute deviation
        return np.nanmedian(np.abs(x - np.nanmedian(x)))
    
    def iqr(x):
        #inter quantile range
        return np.diff(np.nanpercentile(x, [25, 75]))[0]
    
    
    window_minutes = 10
    
    window_frames = int(60*window_minutes*fps)
    
    df['timestamp_m'] = (df['timestamp']/window_frames).round().astype(np.int)
    val = feat_columns + ['timestamp_m'] #only aggregate the valid columns
    dat_agg = df[val].groupby('timestamp_m').agg(['median', top10, top90, mad, iqr])
    return dat_agg

if __name__ == '__main__':

    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    dname = '/Volumes/behavgenom_archive$/Solveig/All/Results/'
    
    fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive = True)
    
    stat_data = {}
    for ifname, fname in enumerate(fnames):
        print(ifname+1, len(fnames))
        
        exp_key = fname.split(os.sep)[-2].split('_')[-1]
        
        
        with pd.HDFStore(fname, 'r') as fid:
            timeseries_features = fid['/timeseries_features']
        
        fps = read_fps(fname)
        dat = averages_by_time(timeseries_features, fps)
        
        if not exp_key in stat_data:
            stat_data[exp_key] = []
        stat_data[exp_key].append(dat)
    #%%
    feats = ['angular_velocity', 'area', 'area_length_ratio', 'curvature_head',
       'curvature_hips', 'curvature_midbody', 'curvature_neck',
       'curvature_tail', 'eigen_projection_1', 'eigen_projection_2',
       'eigen_projection_3', 'eigen_projection_4', 'eigen_projection_5',
       'eigen_projection_6', 'eigen_projection_7', 'head_tail_distance',
       'length', 'major_axis', 'minor_axis', 'quirkiness',
       'relative_angular_velocity_head_tip', 'relative_angular_velocity_hips',
       'relative_angular_velocity_neck', 'relative_angular_velocity_tail_tip',
       'relative_radial_velocity_head_tip', 'relative_radial_velocity_hips',
       'relative_radial_velocity_neck', 'relative_radial_velocity_tail_tip',
       'relative_speed_midbody', 'speed', 'width_head_base',
       'width_length_ratio', 'width_midbody', 'width_tail_base']
    
    stats = ['median', 'top10', 'top90', 'mad', 'iqr']
    
    n_exp = 2
    
    cohort_dat = []
    for n_cohort in range(1, 4):
        exp_key = 'exp{}co{}'.format(n_exp, n_cohort)
        
        feat_nn = {}
        for feat in feats:
            feat_nn[feat] = {}
            for stat in stats:
                dd = [x[feat][stat] for x in stat_data[exp_key]]
                
                feat_nn[feat][stat] = np.mean(dd, axis=0)
        cohort_dat.append(feat_nn)
    #%%
    strC = 'bgr'
    for feat in feats:
        plt.figure()
        ax1 = None
        n_rows = len(stats)
        
        for istat, stat in enumerate(stats):
            if ax1 is None:
                ax1 = plt.subplot(n_rows, 1, istat+1)
            else:
                if stat not in ['mad', 'iqr']:
                    plt.subplot(n_rows, 1, istat+1, sharex=ax1, sharey=ax1)
                else:
                    plt.subplot(n_rows, 1, istat+1, sharex=ax1)
                    
            for ii in range(3):
                plt.plot(cohort_dat[ii][feat][stat], c = strC[ii])
                plt.title(stat)
        plt.suptitle(feat)
    
        #%%
#   fig, axes = plt.subplots(nrows=2, ncols=1)
#        cols2keep = ['worm_index', 'timestamp', 'speed', 'length']
#        timeseries_features = timeseries_features[['worm_index', 'timestamp', 'speed', 'length']]
#        
#        #%%
#        stat_colors = {'median':'b', 'top10':'r', 'top90':'g'}
#        
#        dat = averages_by_time(timeseries_features)
#        for ii, feat in enumerate(feat_columns):
#            for stat in stat_colors: 
#                plt.sca(axes[ii])
#                plt.plot(dat[feat][stat], c=stat_colors[stat])
#                plt.title(feat)

