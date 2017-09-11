#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""

import tables
import numpy as np
import matplotlib.path as mplPath
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from tierpsy.helper.params import read_microns_per_pixel
from tierpsy.helper.misc import TABLE_FILTERS, remove_ext, \
TimeCounter, print_flush, get_base_name

from tierpsy_features.helper import DataPartition
#%%
def _is_valid_cnt(x):
    return x is not None and \
           x.size >= 2 and \
           x.ndim ==2 and \
           x.shape[1] == 2

def _h_smooth_cnt(food_cnt, resampling_N = 1000, smooth_window=None, _is_debug=False):
    if smooth_window is None:
        smooth_window = resampling_N//20
    
    if not _is_valid_cnt(food_cnt):
        #invalid contour arrays
        return food_cnt
        
    smooth_window = smooth_window if smooth_window%2 == 1 else smooth_window+1
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(food_cnt[:, 0])
    dy = np.diff(food_cnt[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)
    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]
    fx = interp1d(lengths, food_cnt[:, 0])
    fy = interp1d(lengths, food_cnt[:, 1])
    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)
    
    rx = fx(subLengths)
    ry = fy(subLengths)
    
    pol_degree = 3
    rx = savgol_filter(rx, smooth_window, pol_degree)
    ry = savgol_filter(ry, smooth_window, pol_degree)
    
    food_cnt_s = np.stack((rx, ry), axis=1)
    
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(food_cnt[:, 0], food_cnt[:, 1], '.-')
        plt.plot(food_cnt_s[:, 0], food_cnt_s[:, 1], '.-')
        plt.axis('equal')
        plt.title('smoothed contour')
    
    return food_cnt_s

def _h_get_closest_index(obj_coords, contour_coords):
    rr = np.linalg.norm(obj_coords - contour_coords, axis=1)
    ind_peak = np.argmin(rr)
    ind_dist = rr[ind_peak]
    return ind_peak, ind_dist

def get_cnt_feats(skeletons, 
                  food_cnt,
                  _is_debug = False):
    #%%
    partitions = ['head_base', 'tail_base', 'midbody']
    p_obj = DataPartition(partitions, n_segments=skeletons.shape[1])
    skel_avg = p_obj.apply_partitions(skeletons, func=np.mean)
    #%%
    midbody_cc = skel_avg['midbody']
    rr = np.linalg.norm(midbody_cc[:, None, :] - food_cnt[None, ...], axis=2)
    ind_peak = np.argmin(rr, axis=1)
    ind_dist = np.array([x[i] for i,x in zip(ind_peak, rr)])
    
    cnt_ind, dist_from_cnt = map(np.array, zip(*map(_h_get_closest_index, midbody_cc)))
    
    #find if the trajectory points are inside the closed polygon (outside will be negative)
    bbPath = mplPath.Path(food_cnt)
    outside = ~bbPath.contains_points(midbody_cc)
    dist_from_cnt[outside] = -dist_from_cnt[outside]
    
    get_unit_vec = lambda x : x/np.linalg.norm(x, axis=1)[:, np.newaxis]
    
    top = cnt_ind+1
    top[top>=food_cnt.shape[0]] -= food_cnt.shape[0] #fix any overflow index
    bot = cnt_ind-1 #it is not necessary to correct because we can use negative indexing
    
    food_u =  get_unit_vec(food_cnt[top]-food_cnt[bot])
    worm_u = get_unit_vec(skel_avg['head_base'] - skel_avg['tail_base'])
    
    dot_prod = np.sum(food_u*worm_u, axis=1)
    orientation_food_cnt = np.arccos(dot_prod)*180/np.pi
    #%%
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure(figsize=(12,12))
        
        plt.subplot(2,2,2)
        plt.plot(orientation_food_cnt)
        plt.title('Orientation respect to the food contour')
        
        plt.subplot(2,2,4)
        plt.plot(dist_from_cnt)
        plt.title('Distance from the food contour')
        
        plt.subplot(1,2,1)
        plt.plot(food_cnt[:,0], food_cnt[:,1])
        plt.plot(midbody_cc[:,0], midbody_cc[:,1], '.')
        plt.plot(food_cnt[cnt_ind,0], food_cnt[cnt_ind,1], 'r.')
        plt.axis('equal')
    #%%  
    
    return orientation_food_cnt, dist_from_cnt, cnt_ind

#%%
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
    #%%
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))

def read_food_contour(skeletons_file):
    try:
        with tables.File(skeletons_file, 'r') as fid:
            food_cnt_pix = fid.get_node('/food_cnt_coord')[:]
            
        #smooth contours
        microns_per_pixel = read_microns_per_pixel(skeletons_file)
        food_cnt = microns_per_pixel*food_cnt_pix
        food_cnt = _h_smooth_cnt(food_cnt)
        
    except tables.exceptions.NoSuchNodeError:
        food_cnt = None
    
    
    
    
    return food_cnt

#%%
if __name__ == '__main__':
    import pandas as pd
    
    skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/_TEST/N2_worms10_CSCD563206_10_Set9_Pos4_Ch6_25072017_214236_skeletons.hdf5'
    features_file = skeletons_file.replace('_skeletons.hdf5', '_featuresN.hdf5')
    
    food_cnt = read_food_contour(skeletons_file)

    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    trajectories_data = trajectories_data[trajectories_data['skeleton_id']>-1]
    
    for worm_index, worm_data in trajectories_data.groupby('worm_index_joined'):
        skel_id = worm_data['skeleton_id'].values 
        with tables.File(features_file, 'r') as fid:
            skeletons = fid.get_node('/coordinates/skeletons')[skel_id, :, :]
    
        outputs = get_cnt_feats(skeletons, 
                  food_cnt,
                  _is_debug = False)