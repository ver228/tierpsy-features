#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""

#%%
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pylab as plt
    import tables
    import numpy as np
    
    from tierpsy.helper.params import read_fps
    from tierpsy.analysis.feat_init.smooth_skeletons_table import read_food_contour
    from tierpsy_features import get_timeseries_features, all_columns

    #skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/N2_worms10_CSCD563206_10_Set9_Pos4_Ch6_25072017_214236_skeletons.hdf5'
    #skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/N2_worms10_CSCD438313_10_Set12_Pos5_Ch4_25072017_223347_skeletons.hdf5'
    skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/MY23_worms5_food1-10_Set4_Pos5_Ch4_29062017_140148_skeletons.hdf5'
    #skeletons_file = '/Volumes/behavgenom_archive$/Solveig/Results/Experiment8/170822_matdeve_exp8co3/170822_matdeve_exp8co3_12_Set0_Pos0_Ch1_22082017_140000_skeletons.hdf5'
    
    features_file = skeletons_file.replace('_skeletons.hdf5', '_featuresN.hdf5')
    
    fps = read_fps(features_file)    
    food_cnt = read_food_contour(skeletons_file)
    
    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    trajectories_data = trajectories_data[trajectories_data['skeleton_id']>-1]
    
    for worm_index, worm_data in trajectories_data.groupby('worm_index_joined'):
        skel_id = worm_data['skeleton_id'].values 
        #%%
        with tables.File(features_file, 'r') as fid:
            skeletons = fid.get_node('/coordinates/skeletons')[skel_id, :, :]
            widths = fid.get_node('/coordinates/widths')[skel_id, :]
            dorsal_contours = fid.get_node('/coordinates/dorsal_contours')[skel_id, :]
            ventral_contours = fid.get_node('/coordinates/ventral_contours')[skel_id, :]
           #%% 
           
        edging_limit = np.nanmedian(np.max(widths, axis=1))*2
           
        if np.sum(~np.isnan(skeletons[: ,0, 0])) < 200:
            continue
        
        
        outputs = get_timeseries_features(skeletons, 
                                          widths, 
                                          dorsal_contours, 
                                          ventral_contours,
                                          fps,
                                          food_cnt
                                          )

        break