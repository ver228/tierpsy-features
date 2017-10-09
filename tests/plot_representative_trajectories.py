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
import tables
import matplotlib.pylab as plt
import seaborn as sns
from tierpsy_features.summary_stats import get_time_groups
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    all_fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive = True)
    
    set_prefix = '4h_N'
    exp_valid = [7,8]
    time_ranges_m = [(n*15, (n+1)*15) for n in range(8)]
    
    feats_data = pd.read_csv(set_prefix + '_data.csv')
    
    feat_g = feats_data.groupby(('cohort_n', 'time_group'))
    med_values = feat_g.agg('median')
    
    #%%
    time_groups = [0.0, 105.0]
    
    #feat = 'motion_mode_paused_fraction'
    feat = 'speed_norm_90th'
    save_name = 'trajectories_{}.pdf'.format(feat)
    with PdfPages(save_name) as pdf_pages:
        for cohort_n in [1,2,3]:
            for time_group in time_groups[:1]:
                print(cohort_n, time_group)
                gg = feat_g.get_group((cohort_n, time_group))
                
                dd = np.abs(gg[feat]-med_values[feat][cohort_n][time_group])
                fname = gg.loc[dd.argmin(), 'fname']
                print(fname)
                
                with pd.HDFStore(fname, 'r') as fid:
                    fps = fid.get_storer('/trajectories_data').attrs['fps']
                    df = fid['/timeseries_features']
                df['time_group'] = get_time_groups(df['timestamp'], time_ranges_m, fps)
                df_g = df.groupby('time_group')
                
                plt.figure(figsize=(12,6))
                for ii, tg2 in enumerate(time_groups):
                    df_t = df_g.get_group(tg2)
                    
                    plt.subplot(1,2, ii+1)
                    with tables.File(fname, 'r') as fid:
                        food_cnt_coord = fid.get_node('/food_cnt_coord')[:]
                        plt.plot(food_cnt_coord[:, 1], food_cnt_coord[:, 0], 'r', lw=3.0)
                        
                        w_g= df_t.groupby('worm_index')
                        cols = sns.color_palette("hls", len(w_g.groups))
                        for c, (w_ind, w_dat) in zip(cols, w_g):
                            skels = fid.get_node('/coordinates/skeletons')[w_dat.index[::50]]
                            skels = skels.T
                            
                            plt.plot(skels[1], skels[0], color=c)
                            
                    plt.axis('equal')
                    plt.xlim((0, 18000))
                    plt.ylim((0, 20000)) 
                    #
                    plt.title('Time : {} - {}'.format(int(tg2), int(tg2+15)))
                plt.suptitle('Cohort {}'.format(cohort_n))
                pdf_pages.savefig()
                plt.close()
                