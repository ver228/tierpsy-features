#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:24:25 2017

@author: ajaver
"""

import pandas as pd
import numpy as np
from tierpsy_features.helper import get_n_worms_estimate, get_delta_in_frames, add_derivatives

from tierpsy_features.events import get_event_stats, event_region_labels
from tierpsy_features.path import get_path_extent_stats
from tierpsy_features import timeseries_feats_columns, \
ventral_signed_columns, path_curvature_columns, curvature_columns


 #this is a hack to do not calculate the paused subdivision. I does not seem to be informative at all...
event_region_labels_r = event_region_labels.copy()
#del event_region_labels_r['motion_mode'][0]


blob_feats_columns = ['blob_area',
 'blob_perimeter',
 'blob_box_length',
 'blob_box_width',
 'blob_quirkiness',
 'blob_compactness',
 'blob_solidity',
 'blob_hu0',
 'blob_hu1',
 'blob_hu2',
 'blob_hu3',
 'blob_hu4',
 'blob_hu5',
 'blob_hu6']

feats2normalize = {
    'L' : [
       'head_tail_distance',
       'major_axis', 
       'minor_axis', 
       'dist_from_food_edge',
       'length',
       'width_head_base', 
       'width_midbody', 
       'width_tail_base'
       ],
    '1/L' : path_curvature_columns + curvature_columns,
    'L^2' : ['area']
}
feats2normalize['L'] += [x for x in timeseries_feats_columns if 'radial_velocity' in x]
feats2normalize['L'] += [x for x in timeseries_feats_columns if 'speed' in x]
#%%
def _normalize_by_w_length(timeseries_data, feats2norm):
    '''
    Normalize features by body length. This is far from being the most efficient solution, but it is the easier to implement.
    '''
    
    def _get_conversion_vec(units_t, median_length_vec):
        '''helper function to find how to make the conversion'''
        if units_t == 'L':
            conversion_vec = 1/median_length_vec
        elif units_t == '1/L':
            conversion_vec = median_length_vec
        elif units_t == 'L^2':
            conversion_vec = median_length_vec**2
        return conversion_vec
    
    timeseries_data = timeseries_data.copy()
    
    median_length = timeseries_data.groupby('worm_index').agg({'length':'median'})
    median_length_vec = timeseries_data['worm_index'].map(median_length['length'])
    
    changed_feats_l = []
    for units_t, feats in feats2norm.items():
        feats_f = [x for x in timeseries_data if any(x.startswith(f) for f in feats)]
        conversion_vec = _get_conversion_vec(units_t, median_length_vec)
        for f in feats_f:
            timeseries_data[f] *= conversion_vec
        changed_feats_l += feats_f

    changed_feats = {x: x + '_norm' for x in changed_feats_l}    
    timeseries_data = timeseries_data.rename(columns = changed_feats)
    
    return timeseries_data, changed_feats

def get_df_quantiles(df,
                     feats2check = timeseries_feats_columns,
                     subdivision_dict = {'food_region':['orientation_food_edge']},
                     feats2norm = feats2normalize,
                     feats2abs = ventral_signed_columns,
                     is_remove_subdivided = True,
                     is_abs_ventral = True,
                     is_normalize = False
                     ):
    '''
    Get quantile statistics for all the features given by `feats2check`.
    In the features in `feats2abs` we are going to use only the absolute. This is to 
    deal with worms with unknown dorsal/ventral orientation.
    '''
    q_vals = [0.1, 0.5, 0.9] #percentiles to calculate
    iqr_limits = [0.25, 0.75] # range of percentiles used for the interquantile distance
    valid_q = q_vals + iqr_limits
    
    df = df.copy() #like this i can modify directoy the df without long lasting consequences
    
    #filter features to be abs
    def _filter_ventral_features(feats2check):#%%
        valid_f = [x for x in feats2check if any(x.startswith(f) for f in feats2abs)]
        return valid_f
    
    #filter default columns in case they are not present
    feats2check = [x for x in feats2check if x in df]
    
    #filter default columns in case they are not present. Same for the subdivision dictionary.
    subdivision_dict_r = {}
    for e_subdivide, feats2subdivide in subdivision_dict.items():
        ff = [x for x in feats2check if x in feats2subdivide]
        if e_subdivide in df and ff:
            subdivision_dict_r[e_subdivide] = ff
    subdivision_dict = subdivision_dict_r
    
    
    #subdivide a feature using the event features
    subdivided_df = _get_subdivided_features(df, subdivision_dict = subdivision_dict)
    df = df.join(subdivided_df)
    feats2check += subdivided_df.columns.tolist()
    if is_remove_subdivided:
        df = df[[x for x  in df if not x in feats2subdivide]]
        feats2check = [x for x in feats2check if x not in feats2subdivide]
    
    #add normalized features
    if is_normalize:
        df, changed_feats = _normalize_by_w_length(df, feats2norm = feats2norm)
        feats2check = [x if not x in changed_feats else changed_feats[x] for x in feats2check]
    
    #abs features that are ventral/dorsal side
    if is_abs_ventral:
        feats2abs = _filter_ventral_features(feats2check)
        #find features that match ventral_signed_columns
        if feats2abs:
            #normalize
            df[feats2abs] = df[feats2abs].abs()
            #change name
            df.columns = [x + '_abs' if x in feats2abs else x for x in df.columns]
            feats2check = [x + '_abs' if x in feats2abs else x for x in feats2check]
    
    #calculate quantiles
    feat_mean = None
    Q = df[feats2check].quantile(valid_q)
    feat_mean = pd.concat((feat_mean, Q), axis=1)

    #name correctly
    dat = []
    for q in q_vals:
        q_dat = feat_mean.loc[q]
        q_str = '_{}th'.format(int(round(q*100)))
        for feat, val in q_dat.iteritems():
            dat.append((val, feat+q_str))
    
    
    IQR = feat_mean.loc[0.75] - feat_mean.loc[0.25]
    dat += [(val, feat + '_IQR') for feat, val in IQR.iteritems()]
    
    feat_mean_s = pd.Series(*list(zip(*dat)))
    return feat_mean_s


def _get_subdivided_features(timeseries_data, subdivision_dict):
    '''
    subdivision_dict = {event_v1: [feature_v1, feature_v2, ...], event_v2: [feature_vn ...], ...}
    
    event_vector = [-1, -1, 0, 0, 1, 1]
    feature_vector = [1, 3, 4, 5, 6, 6]
    
    new_vectors ->
    [1, 3, nan, nan, nan, nan]
    [nan, nan, 4, 5, nan, nan]
    [nan, nan, nan, nan, 6, 6]
    
    '''
    
    #assert all the subdivision keys are known events
    assert all(x in event_region_labels_r.keys() for x in subdivision_dict)
    
    
    event_type_link = {#%%
            'food_region' : '_in_',
            'motion_mode' : '_w_'
            }
    subdivided_data = []
    for e_col, timeseries_cols in subdivision_dict.items():
        e_data = timeseries_data[e_col].values
        
        if e_col in event_type_link:
            str_l = event_type_link[e_col]
        else:
            str_l = '_'
        
        for flag, label in event_region_labels[e_col].items():
            _flag = e_data != flag
            
            for f_col in timeseries_cols:
                f_data = timeseries_data[f_col].values.copy()
                
                try:
                    f_data[_flag] = np.nan
                except:
                    import pdb
                    pdb.set_trace()
                new_name = f_col + str_l + label
                
                subdivided_data.append((new_name, f_data))
    
    
    if not subdivided_data:
        #return empty df if nothing was subdivided
        return pd.DataFrame([])
    

    columns, data = zip(*subdivided_data)
    subdivided_df = pd.DataFrame(np.array(data).T, columns = columns)
    subdivided_df.index = timeseries_data.index

    return subdivided_df



#%%
def get_summary_stats(timeseries_data, 
                      fps, 
                      blob_features = None, 
                      derivate_delta_time = None,
                      only_abs_ventral = False):
    
    #TODO I need to decided a clever way to add the derivatives
    ts_cols_all = timeseries_feats_columns + ['d_' + x for x in timeseries_feats_columns]
    v_sign_cols = ventral_signed_columns + ['d_' + x for x in ventral_signed_columns]
    
    feats2norm = {}
    for k,dat in feats2normalize.items():
        feats2norm[k] = dat + ['d_' + x for x in dat]
    ts_cols_norm = sum(feats2norm.values(), [])
    
    #summarize everything
    exp_feats = [] 
    
    ## event features
    n_worms_estimate = get_n_worms_estimate(timeseries_data['timestamp'])
    event_stats_s = get_event_stats(timeseries_data, fps , n_worms_estimate)
    
    ## timeseries features
    
    ##### simple
    timeseries_stats_s = get_df_quantiles(timeseries_data, 
                                          feats2check = ts_cols_all,
                                          feats2abs = v_sign_cols,
                                          feats2norm = feats2norm, 
                                          is_normalize = False)
    
    path_grid_stats_s = get_path_extent_stats(timeseries_data, fps, is_normalized = False)
    
    feat_stats = pd.concat((timeseries_stats_s, path_grid_stats_s, event_stats_s))
    
    exp_feats.append(feat_stats)
    ##### normalized by worm length
    timeseries_stats_n = get_df_quantiles(timeseries_data, 
                                          feats2check = ts_cols_norm,
                                          feats2abs = v_sign_cols,
                                          feats2norm = feats2norm, 
                                          is_normalize = True)
    
    path_grid_stats_n = get_path_extent_stats(timeseries_data, fps, is_normalized = True)
    feat_stats_n = pd.concat((timeseries_stats_n, path_grid_stats_n))
    exp_feats.append(feat_stats_n)
    
    #add subdivisions
    feat_stats_m_subdiv = get_df_quantiles(timeseries_data, 
                                      feats2check = ts_cols_all, 
                                      feats2abs = v_sign_cols,
                                      feats2norm = feats2norm, 
                                      subdivision_dict = {'motion_mode' : ts_cols_all}, 
                                      is_abs_ventral = True) #i only calculate the subdivision abs or not abs
    exp_feats.append(feat_stats_m_subdiv)
    
    if not only_abs_ventral:
        ##### non-abs ventral signed features
        feat_stats_v = get_df_quantiles(timeseries_data, 
                                              feats2check = v_sign_cols,
                                              feats2abs = v_sign_cols,
                                              feats2norm = feats2norm, 
                                              is_abs_ventral = False,
                                              is_normalize = False)
        
        exp_feats.append(feat_stats_v)
        
        ##### non-abs and normalized ventral signed features
        v_sign_cols_norm = list(set(v_sign_cols) & set(ts_cols_norm))
        feat_stats_v_n = get_df_quantiles(timeseries_data, 
                                              feats2check = v_sign_cols_norm,
                                              feats2abs = v_sign_cols,
                                              feats2norm = feats2norm,
                                              is_abs_ventral = False,
                                              is_normalize = True)
        exp_feats.append(feat_stats_v_n)
    
        
        #add subdivisions
        feat_stats_m_subdiv_v = get_df_quantiles(timeseries_data, 
                                      feats2check = v_sign_cols, 
                                      feats2abs = v_sign_cols,
                                      subdivision_dict = {'motion_mode' : ts_cols_all}, 
                                      is_abs_ventral = False,
                                      is_normalize = False) #i only calculate the subdivision abs or not abs
    
        exp_feats.append(feat_stats_m_subdiv_v)
    
    
    if blob_features is not None:
        assert not ((blob_features is None) and (delta_frames is None))
        index_cols = ['worm_index', 'timestamp']
        
        #add the blob prefix to the blob features if it is not present
        filt_func = lambda x : (not x.startswith('blob_') or (x in index_cols))
        blob_features.columns = ['blob_' + x if filt_func(x) else x for x in blob_features.columns ]
        
        #add blob derivatives
        
        derivate_delta_frames = get_delta_in_frames(derivate_delta_time, fps)
        #TODO I need to decided a clever way to add the derivatives
        blob_features = pd.concat((timeseries_data[index_cols], blob_features), axis=1)
        blob_l = []
        for w_ind, blob_w in blob_features.groupby('worm_index'):
            blob_w = add_derivatives(blob_w, blob_feats_columns, derivate_delta_frames, fps)
            blob_l.append(blob_w)

        if blob_l:
            blob_features = pd.concat(blob_l, axis=0)
            
            #select only the valid columns
            blob_feats_columns_d = blob_feats_columns + ['d_' + x for x in blob_feats_columns]
            blob_cols = [x for x in blob_feats_columns_d if x in blob_features]
            blob_features = blob_features[blob_cols]
            
            #get blobstats
            blob_stats = get_df_quantiles(blob_features, feats2check = blob_cols)
            
            blob_features['motion_mode'] = timeseries_data['motion_mode']
            blob_stats_m_subdiv = get_df_quantiles(blob_features, 
                                              feats2check = blob_cols, 
                                              subdivision_dict = {'motion_mode':blob_cols}, 
                                              is_abs_ventral = False)
            exp_feats += [blob_stats, blob_stats_m_subdiv] 
                       
    exp_feats = pd.concat(exp_feats)
    return exp_feats
    
def _get_time_groups(timestamp, time_ranges_m, fps):
    '''
    Obtain a vector that divides the timeseries data into groups
    
    time_ranges_m - it can be either: 
        a) None : the function will return an all zeros vector
        b) int or float : the timestamp will be binned using this value (in minutes).
        c) list of pairs : the ranges given in this list will be used.
    
    The values returned in time_group corresponds to the initial time in minutes of the group.
    
    '''
    if time_ranges_m is None:
        time_group = np.zeros_like(timestamp)
    elif isinstance(time_ranges_m, (float, int)):
        # a number is given, we can use this faster method to create uniform bins
        window_frames = int(60*time_ranges_m*fps)
        time_group = np.floor(timestamp/window_frames).astype(np.int)
        time_group *= time_ranges_m
    else:
        #add a flag for each of the given regions
        fpm = fps*60
        time_ranges = [(x[0]*fpm, x[1]*fpm) for x in time_ranges_m]
        time_group = np.full_like(timestamp, -1.)
        for t0, tf in time_ranges:
            good = (timestamp >= t0) & (timestamp <= tf)
            time_group[good] = t0/fpm
    
    return time_group

def get_summary_stats_binned(df, fps, time_ranges_m, is_normalize):
    '''
    Calculate the stats of the features in df using `get_feat_stats` by binning 
    the data according to its time. See `_get_time_groups` to see the kind of 
    values that `time_range_m` accepts.
    '''
    df['time_group'] = _get_time_groups(df['timestamp'], time_ranges_m, fps)
    dat_agg = []
    for ind_t, dat in df.groupby('time_group'):
        if ind_t< 0:
            continue
        
        feat_means_s = get_summary_stats(dat, fps, is_normalize) 
        
        #add the time information at the begining
        feat_means_s = pd.Series(ind_t, index=['time_group']).append(feat_means_s)
        
        dat_agg.append(feat_means_s)
    
    dat_agg = pd.concat(dat_agg, axis=1).T
    return dat_agg

def collect_summary_stats(fnames, info_df, time_ranges_m = None, is_normalize = False):
    reserved_w = ['time_group', 'worm_index', 'timestamp']
    assert not any(x in info_df for x in reserved_w)
    
    all_data = []
    for (iexp, info_row), fname in zip(info_df.iterrows(), fnames):
        print(iexp + 1, len(fnames))
        
        with pd.HDFStore(fname, 'r') as fid:
            fps = fid.get_storer('/trajectories_data').attrs['fps']
            features_timeseries = fid['/timeseries_data']
            
        feat_mean_df = get_summary_stats_binned(features_timeseries, 
                                           fps, 
                                           time_ranges_m,
                                           is_normalize
                                           )
        
        #add the info to the feat_means
        dd = pd.concat([info_row] * feat_mean_df.shape[0], axis=1).T
        dd.index = feat_mean_df.index
        feat_mean_df = dd.join(feat_mean_df)
        
        all_data.append(feat_mean_df)
        
    feat_means_df = pd.concat(all_data, ignore_index=True)
    return feat_means_df


#%%
if __name__ == '__main__':
    from tierpsy.helper.params import read_fps
    #fname = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3_featuresN.hdf5'
    #%%
    
    fname = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_020617/WN2002_worms10_food1-10_Set1_Pos4_Ch4_02062017_115723_featuresN.hdf5'
    with pd.HDFStore(fname, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features']
    fps = read_fps(fname)
    delta_frames = max(1, int(fps/3))
    
    
    feat_stats = get_summary_stats(timeseries_data, 
                                   fps,  
                                   blob_features, 
                                   delta_frames,
                                   only_abs_ventral = True
                                   )
    