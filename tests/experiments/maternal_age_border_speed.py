#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:20:33 2017

@author: ajaver
"""

import pandas as pd
import os
import glob
import numpy as np
from tierpsy_features.summary_stats import get_time_groups, events_from_df


if __name__ == '__main__':
    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    all_fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive = True)
    
    set_prefix = '4h_N'
    exp_valid = [7,8]
    time_ranges_m = [(0, 15), (15, 30), (30, 60), (60, 120)]
    #[(n*15, (n+1)*15) for n in range(8)]
    
    feats_data = pd.read_csv(set_prefix + '_data.csv')
    feats_data = feats_data.drop_duplicates(subset='fname')
    
    all_speeds = []
    for nn_i, (irow, row) in enumerate(feats_data.iterrows()):
        print(nn_i+1, feats_data.shape[0])
        fname = row['fname']
        #%%
        with pd.HDFStore(fname, 'r') as fid:
            fps = fid.get_storer('/trajectories_data').attrs['fps']
            features_timeseries = fid['/timeseries_features']
        event_df, event_durations = events_from_df(features_timeseries, fps)
        event_durations['timestamp_initial'] = event_durations['timestamp_initial'].astype(np.int)
        
        good = (event_durations['event_type'] == 'food_region') & \
            (event_durations['region'] == 0) & \
            (event_durations['edge_flag'] >= 0)
        border_events = event_durations[good]
        #%%
        delT = int(15*fps)
        n_seg = delT*2 + 1
        #i eliminate the border of the table to avoid complications.
        #not very elegant but it is the easier
        ts_w = features_timeseries.groupby('worm_index')
        for _, b_row in border_events.iterrows():
            w_data = ts_w.get_group(b_row['worm_index'])
            f_b = w_data[w_data['timestamp'] == b_row['timestamp_initial']]
            
            
            ini = f_b.index[0] - delT
            fin = f_b.index[0] + delT
            
            speed = w_data.loc[ini:fin, 'speed'].copy()
            speed = speed/w_data['length'].median()
            
            #eliminate any data i could have peak form the borders
            pad_n = (max(0, speed.index[0]-ini), max(0, fin-speed.index[-1]))
            speed = np.pad(speed.values, pad_n, 'constant', constant_values=np.nan)
            
            dist_from_food_edge = w_data.loc[ini:fin, 'dist_from_food_edge'].copy()
            dist_from_food_edge = np.pad(dist_from_food_edge.values, pad_n, 'constant', constant_values=np.nan)
            
            assert speed.size == n_seg
            
            b_row_i = b_row.copy()
            b_row_i['iexp'] = irow
            all_speeds.append((b_row_i, speed, dist_from_food_edge))
        
    #%%
    import matplotlib.pylab as plt
    
    #border_events_df, all_speeds_df, all_dist_df = zip(*all_speeds)
    border_events_df, all_speeds_d = zip(*all_speeds)
    border_events_df = pd.concat(border_events_df, axis=1, ignore_index=True).T
    all_speeds_d = np.vstack(all_speeds_d)
    #%%
    border_events_df['time_group'] = get_time_groups(border_events_df['timestamp_initial'], time_ranges_m, fps)
    info_df = feats_data[['exp_n', 'cohort_n']].copy()
    info_df['iexp'] = info_df.index
    border_events_df = pd.merge(border_events_df, info_df, on='iexp')
    #%%
    import seaborn as sns
    
    plt.figure(figsize = (15, 12))
    change_avgs = {}
    b_event_g = border_events_df.groupby(('cohort_n', 'time_group'))
    valid_keys = sorted([x for x in b_event_g.groups.keys() if x[1] >= 0])
    for iplot, (cohort_n, time_group) in enumerate(valid_keys):
        dat = b_event_g.get_group((cohort_n, time_group))
        
        plt.subplot(3, len(time_ranges_m), iplot+1)
        s = all_speeds_d[dat.index]
        s = np.abs(s)
        b_p = delT- 100
        s = s[:, 100:651]
        s = s[~np.any(np.isnan(s), axis=1)]
        sm = np.nanmean(s, axis=0)
        plt.plot(s.T, 'gray')
        plt.plot(sm, lw=3)
        plt.plot((b_p, b_p), plt.ylim(), ':r')
        plt.title('C{}_T{}'.format(cohort_n, time_group))
        plt.ylim((-0.01, 0.5))
        
        serr = np.std(s, axis=0)/np.sqrt(s.shape[0])
        change_avgs[(cohort_n, time_group)] = (sm, serr)
    #%%
    time_groups = [x[1] for x in change_avgs.keys() if x[0]==1]
    
    for time_group in time_groups:
        plt.figure(figsize = (12, 5))
        for cohort_n in [1,2,3]:
            sm, serr = change_avgs[(cohort_n, time_group)]
            plt.errorbar(np.arange(sm.size), sm, yerr=serr)
        plt.plot((b_p, b_p), plt.ylim(), ':k')
        plt.title('T{}'.format(time_group))
    
    
    