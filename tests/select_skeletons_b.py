#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:37:18 2017

@author: ajaver
"""
import pandas as pd
import tables
import glob
import os

from tierpsy.analysis.feat_tierpsy.obtain_tierpsy_features import _h_get_timeseries_feats_table

if __name__ == '__main__':
    
    dname = '/Volumes/behavgenom_archive$/Solveig/All/Results/'
    
    fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive=True)
    
    for ifname, fname in enumerate(fnames):
        print(ifname + 1, len(fnames))
        _h_get_timeseries_feats_table(fname, delta_time=1/3, curvature_window=1)
    #%%
#    with pd.HDFStore(fname, 'r') as fid:
#        timeseries_features = fid['/timeseries_features']
#    #%%
#    
#    dd = timeseries_features['worm_index'] == 721
#    
    
    
    #get_curvature(skeletons, method = 'angle', points_window=None