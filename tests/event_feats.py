#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:41:10 2017

@author: ajaver
"""

import numpy as np
import pandas as pd


def _get_pulses_indexes(light_on, min_window_size=0):
    switches = np.diff(light_on.astype(np.int))
    turn_on, = np.where(switches==1)
    turn_off, = np.where(switches==-1)
    
    assert turn_on.size == turn_off.size
    
    delP = turn_off - turn_on
    
    good = delP > min_window_size
    
    return turn_on[good], turn_off[good]

def _range_vec(vec, th):
    flags = np.zeros(vec.size)
    _out = vec < -th
    _in = vec > th
    flags[_out] = -1
    flags[_in] = 1
    return flags

def _flag_regions(vec, lower_th, higher_th, smooth_window, min_zero_window):
    vv = pd.Series(vec).fillna(method='ffill').fillna(method='bfill')
    smoothed_vec = vv.rolling(window=smooth_window,center=True).mean()
    
    paused_f = (smoothed_vec > -lower_th) & (smoothed_vec < lower_th)
    flag_modes = _range_vec(smoothed_vec, higher_th)
    
    turn_on, turn_off = _get_pulses_indexes(paused_f, min_zero_window)
    inter_pulses = zip([0] + list(turn_off), list(turn_on) + [paused_f.size-1])
    for ini, fin in inter_pulses:
        dd = np.unique(flag_modes[ini:fin+1])
        dd = [x for x in dd if x != 0]
        if len(dd) == 1:
            flag_modes[ini:fin+1] = dd[0]
        elif len(dd) > 1:
            kk = flag_modes[ini:fin+1]
            kk[kk==0] = np.nan
            kk = pd.Series(kk).fillna(method='ffill').fillna(method='bfill')
            flag_modes[ini:fin+1] = kk
    return flag_modes

if __name__ == '__main__':
    from tierpsy.helper.params import read_fps
    import matplotlib.pylab as plt
    import os
    import glob

    
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    fnames = glob.glob(os.path.join(dname, 'Experiment8', '**', '*_featuresN.hdf5'), recursive = True)
    
    for ifname, fname in enumerate(fnames):
        print(ifname+1, len(fnames))
        with pd.HDFStore(fname, 'r') as fid:
            if '/provenance_tracking/FEAT_TIERPSY' in fid:
                timeseries_features = fid['/timeseries_features']
            else:
                continue
        break
    
    #%%
    
    for w_index in [2]:#, 69, 431, 437, 608]:
        
        good = timeseries_features['worm_index']==w_index
        worm_length = timeseries_features.loc[good, 'length'].median()
        speed = timeseries_features.loc[good, 'speed'].values
        dist_from_food_edge = timeseries_features.loc[good, 'dist_from_food_edge'].values
        
        
        smooth_window_s = 0.5
        min_paused_win_food_s = 1
        min_paused_win_speed_s = 1/3
        
        fps = read_fps(fname)
        pause_th_lower = worm_length*0.025
        pause_th_higher = worm_length*0.05
        
        edge_offset_lower = worm_length/2
        edge_offset_higher = worm_length
        
        w_size = int(round(fps*smooth_window_s))
        smooth_window = w_size if w_size % 2 == 1 else w_size + 1
        
        min_paused_win_speed = fps/min_paused_win_speed_s
        min_paused_win_food = fps/min_paused_win_food_s
    
        motion_modes = _flag_regions(speed, 
                                     pause_th_lower, 
                                     pause_th_higher, 
                                     smooth_window, 
                                     min_paused_win_speed
                                     )    
        
        food_position = _flag_regions(dist_from_food_edge, 
                                     edge_offset_lower, 
                                     edge_offset_higher, 
                                     smooth_window, 
                                     min_paused_win_food
                                     )   
        
        #%%
        plt.figure()
        plt.plot(speed)
        plt.plot(motion_modes*pause_th_higher)
        
        plt.figure()
        plt.plot(dist_from_food_edge)
        plt.plot(food_position*edge_offset_lower)

    #%%
        #angular_velocity = timeseries_features.loc[good, 'angular_velocity'].values
        angular_velocity = timeseries_features['angular_velocity'].values
        smooth_window = int(round(fps*2))  
        
        dd = angular_velocity.copy()
        dd[np.isnan(dd)] = 0
        dd = dd.cumsum()
        
        #smoothed_vec = pd.Series(dd).rolling(window=smooth_window,center=True).mean()
        
        #%%
        plt.figure()
        timeseries_features.loc[good, 'curvature_tail'].plot()
        timeseries_features.loc[good, 'curvature_midbody'].plot()
        timeseries_features.loc[good, 'curvature_head'].plot()
        