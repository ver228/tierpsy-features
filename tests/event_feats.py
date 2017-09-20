#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:41:10 2017

@author: ajaver
"""
import glob
import os
import pandas as pd

if __name__ == '__main__':
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
    
    def _range_vec(vec, th):
        flags = np.zeros(vec.size)
        _out = vec < -th
        _in = vec > th
        flags[_out] = -1
        flags[_in] = 1
        return flags
    
    #%%
    
    import numpy as np
    edge_offset = 400 
    dist_from_food_edge = timeseries_features['dist_from_food_edge']
    
    food_position = _range_vec(dist_from_food_edge, edge_offset)
    
    
    #%%
    good = timeseries_features['worm_index']==2
    worm_length = timeseries_features.loc[good, 'length'].median()
    pause_th = worm_length*0.025
    
    speed = timeseries_features.loc[good, 'speed']
    motion_modes = _range_vec(speed, pause_th)
    #%%
    from tierpsy.helper.params import read_fps
    import matplotlib.pylab as plt
    from tierpsy_features.helper import fillbnan, fillfnan
    
    
    fps = read_fps(fname)
    w_size = int(round(fps))
    w_kernel = np.ones(w_size)/w_size
    
    missing_val = np.isnan(speed)
    vv = fillbnan(fillfnan(speed.values))
    dd = np.convolve(vv, w_kernel, mode='valid')
    
    
    plt.plot(speed)
    plt.plot(plt.xlim(), (pause_th,pause_th))
    plt.plot(plt.xlim(), (-pause_th,-pause_th))
    
    mm = 2*pause_th
    plt.plot(plt.xlim(), (mm,mm))
    plt.plot(plt.xlim(), (-mm,-mm))
    
    plt.plot(dd)