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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.style.use('ggplot')

#%%
def averages_by_time(df, fps, window_minutes = 5):
    #%%
    index_columns = ['worm_index', 'timestamp', 'timestamp_m']
    feat_columns = [x for x in df.columns if not x in index_columns]
    df
    def top10(x):
        return np.nanpercentile(x, 10)
    def top90(x):
        return np.nanpercentile(x, 90)
    
    #median absolute deviation
    #def mad(x):
    #    return np.nanmedian(np.abs(x - np.nanmedian(x)))
    
    def iqr(x):
        #inter quantile range
        return np.diff(np.nanpercentile(x, [25, 75]))[0]
    
    fun2agg = ['median', top10, top90, iqr]
    
    
    window_frames = int(60*window_minutes*fps)
    
    df['timestamp_m'] = (df['timestamp']/window_frames).round().astype(np.int)
    val = feat_columns + ['timestamp_m'] #only aggregate the valid columns
    dat_agg = df[val].groupby('timestamp_m').agg(fun2agg)
    
    #%%
    return dat_agg
#%%
if __name__ == '__main__':

    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    #%%
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    fnames = glob.glob(os.path.join(dname, 'Experiment8', '**', '*_featuresN.hdf5'), recursive = True)
    #for ifname, fname in enumerate(fnames):
    #    with tables.File(fname, 'r+') as fid:
    #        fid.remove_node('/provenance_tracking/FEAT_TIERPSY')
    
    #%%
    stat_data = {}
    for ifname, fname in enumerate(fnames):
        print(ifname+1, len(fnames))
        
        exp_key = fname.split(os.sep)[-2].split('_')[-1]
        
        
        with pd.HDFStore(fname, 'r') as fid:
            if '/provenance_tracking/FEAT_TIERPSY' in fid:
                timeseries_features = fid['/timeseries_features']
            else:
                continue
            
        fps = read_fps(fname)
        dat = averages_by_time(timeseries_features, fps)
        
        if not exp_key in stat_data:
            stat_data[exp_key] = []
        stat_data[exp_key].append(dat)
        
    #%%
    feats = ['speed', 'angular_velocity', 'relative_speed_midbody', 
             'relative_radial_velocity_head_tip', 'relative_angular_velocity_head_tip', 
             'relative_radial_velocity_neck', 'relative_angular_velocity_neck', 
             'relative_radial_velocity_hips', 'relative_angular_velocity_hips', 
             'relative_radial_velocity_tail_tip', 'relative_angular_velocity_tail_tip', 
             'length', 'area', 'area_length_ratio', 'width_length_ratio', 
             'width_head_base', 'width_midbody', 'width_tail_base', 'head_tail_distance', 
             'quirkiness', 'major_axis', 'minor_axis', 
             'eigen_projection_1', 'eigen_projection_2', 'eigen_projection_3', 
             'eigen_projection_4', 'eigen_projection_5', 'eigen_projection_6', 
             'eigen_projection_7', 'curvature_head', 'curvature_hips', 
             'curvature_midbody', 'curvature_neck', 'curvature_tail', 
             'orientation_food_edge', 'dist_from_food_edge']
    
    stats = ['median', 'top10', 'top90', 'iqr'] # 'mad', 
    
    

            
    #%%
    save_dir = '/Users/ajaver/OneDrive - Imperial College London/development'
    
    
    for n_exp in [8]:#range(1, 9):
        
        with PdfPages('{}/exp{}_avg.pdf'.format(save_dir, n_exp)) as pdf_saver:
            cohort_dat = []
            for n_cohort in range(1, 4):
                exp_key = 'exp{}co{}'.format(n_exp, n_cohort)
                
                feat_nn = {}
                for feat in feats:
                    feat_nn[feat] = {}
                    for stat in stats:
                        dd = [x[feat][stat] for x in stat_data[exp_key]]
                        dd = [x for x in dd if not np.isnan(x).all()]
                        
                        feat_nn[feat][stat] = (np.mean(dd, axis=0), np.std(dd, axis=0), len(dd))
                
                cohort_dat.append(feat_nn)
            
            
            strC = 'brg'
            for feat in feats:
                fig = plt.figure()
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
                        yy, ss, nn = cohort_dat[ii][feat][stat]
                        
                        plt.errorbar(np.arange(yy.size), yy, yerr=ss, c = strC[ii])
                        plt.title(stat)
                plt.suptitle(feat)
                
                pdf_saver.savefig()
                plt.close()
            
    
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

