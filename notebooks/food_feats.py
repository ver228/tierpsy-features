#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""

import tables

from tierpsy_features.food import _h_smooth_cnt, get_cnt_feats
from tierpsy_features.features import get_timeseries_features

from tierpsy.helper.params import read_microns_per_pixel
from tierpsy.helper.misc import TABLE_FILTERS, remove_ext, \
TimeCounter, print_flush, get_base_name




def getFoodFeatures(mask_file, 
                    skeletons_file,
                    features_file = None,
                    cnt_method = 'NN',
                    solidity_th=0.98,
                    batch_size = 100000,
                    _is_debug = False
                    ):
    if features_file is None:
        features_file = remove_ext(skeletons_file) + '_featuresN.hdf5'
    
    base_name = get_base_name(mask_file)
    
    progress_timer = TimeCounter('')
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))
    
    food_cnt = calculate_food_cnt(mask_file,  
                                  method=cnt_method, 
                                  solidity_th=solidity_th,
                                  _is_debug=_is_debug)
    microns_per_pixel = read_microns_per_pixel(skeletons_file)
    
    #store contour coordinates in pixels into the skeletons file for visualization purposes
    food_cnt_pix = food_cnt/microns_per_pixel
    with tables.File(skeletons_file, 'r+') as fid:
        if '/food_cnt_coord' in fid:
            fid.remove_node('/food_cnt_coord')
        if _is_valid_cnt(food_cnt):
            tab = fid.create_array('/', 
                                   'food_cnt_coord', 
                                   obj=food_cnt_pix)
            tab._v_attrs['method'] = cnt_method
    
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))
    
    feats_names = ['orient_to_food_cnt', 'dist_from_food_cnt', 'closest_cnt_ind']
    feats_dtypes = [(x, np.float32) for x in feats_names]
    
    with tables.File(skeletons_file, 'r') as fid:
        tot_rows = fid.get_node('/skeleton').shape[0]
        features_df = np.full(tot_rows, np.nan, dtype = feats_dtypes)
        
        if food_cnt.size > 0:
            for ii in range(0, tot_rows, batch_size):
                skeletons = fid.get_node('/skeleton')[ii:ii+batch_size]
                skeletons *= microns_per_pixel
                
                outputs = get_cnt_feats(skeletons, food_cnt, _is_debug = _is_debug)
                for irow, row in enumerate(zip(*outputs)):
                    features_df[irow + ii]  = row

    
    with tables.File(features_file, 'a') as fid: 
        if '/food' in fid:
            fid.remove_node('/food', recursive=True)
        fid.create_group('/', 'food')
        if _is_valid_cnt(food_cnt):
            fid.create_carray(
                    '/food',
                    'cnt_coordinates',
                    obj=food_cnt,
                    filters=TABLE_FILTERS) 
        
        fid.create_table(
                '/food',
                'features',
                obj=features_df,
                filters=TABLE_FILTERS) 
    
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))



#%%
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    #skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/N2_worms10_CSCD563206_10_Set9_Pos4_Ch6_25072017_214236_skeletons.hdf5'
    #skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/N2_worms10_CSCD438313_10_Set12_Pos5_Ch4_25072017_223347_skeletons.hdf5'
    skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/MY23_worms5_food1-10_Set4_Pos5_Ch4_29062017_140148_skeletons.hdf5'
    #skeletons_file = '/Volumes/behavgenom_archive$/Solveig/Results/Experiment8/170822_matdeve_exp8co3/170822_matdeve_exp8co3_12_Set0_Pos0_Ch1_22082017_140000_skeletons.hdf5'
    
    features_file = skeletons_file.replace('_skeletons.hdf5', '_featuresN.hdf5')
    
    food_cnt = read_food_contour(skeletons_file)

    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    trajectories_data = trajectories_data[trajectories_data['skeleton_id']>-1]
    
    for worm_index, worm_data in trajectories_data.groupby('worm_index_joined'):
        skel_id = worm_data['skeleton_id'].values 
        #%%
        with tables.File(features_file, 'r') as fid:
            skeletons = fid.get_node('/coordinates/skeletons')[skel_id, :, :]
            dorsal_contours = fid.get_node('/coordinates/dorsal_contours')[skel_id, :, :]
            ventral_contours = fid.get_node('/coordinates/ventral_contours')[skel_id, :, :]
            widths = fid.get_node('/coordinates/widths')[skel_id, :]
        #%%     
        
        if np.sum(~np.isnan(skeletons[: ,0, 0])) < 200:
            continue
        
        food_df = get_cnt_feats(skeletons, 
                                food_cnt,
                                is_smooth_cnt = False,
                                _is_debug = True)
        
        #%%
        features = get_timeseries_features(skeletons, 
                            widths, 
                            dorsal_contours, 
                            ventral_contours,
                            fps = 25,
                            food_cnt = food_cnt,
                            is_smooth_cnt = False,
                            delta_time = 1/3, #delta time in seconds to calculate the velocity
                            curvature_window = 1
                            )