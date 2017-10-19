#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:24:25 2017

@author: ajaver
"""

import pandas as pd
import numpy as np

from tierpsy_features.helper import get_n_worms_estimate
from tierpsy_features.events import get_event_stats, event_region_labels
from tierpsy_features.path import get_path_extent_stats
from tierpsy_features import timeseries_feats_columns, \
ventral_signed_columns, path_curvature_columns, curvature_columns


feats2normalize = {
    'L' : [
       'speed',
       'relative_speed_midbody', 
       'relative_radial_velocity_head_tip',
       'relative_radial_velocity_neck',
       'relative_radial_velocity_hips',
       'relative_radial_velocity_tail_tip',
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

def _h_get_conversion_vec(units_t, median_length_vec):
    if units_t == 'L':
        conversion_vec = 1/median_length_vec
    elif units_t == '1/L':
        conversion_vec = median_length_vec
    elif units_t == 'L^2':
        conversion_vec = median_length_vec**2
    return conversion_vec
    
def _h_filter_ventral_features(feats2check):
    valid_f = [x for x in feats2check if any(x.startswith(f) for f in ventral_signed_columns)]
    return valid_f

def _normalize_by_w_length(timeseries_data, feats2norm):
    '''
    Normalize features by body length. This is far from being the most efficient solution, but it is the easier to implement.
    '''
    timeseries_data = timeseries_data.copy()
    
    median_length = timeseries_data.groupby('worm_index').agg({'length':'median'})
    median_length_vec = timeseries_data['worm_index'].map(median_length['length'])
    
    changed_feats_l = []
    for units_t, feats in feats2normalize.items():
        feats_f = [x for x in timeseries_data if any(x.startswith(f) for f in feats)]
        conversion_vec = _h_get_conversion_vec(units_t, median_length_vec)
        for f in feats_f:
            timeseries_data[f] *= conversion_vec
        changed_feats_l += feats_f

    changed_feats = {x: x + '_norm' for x in changed_feats_l}    
    timeseries_data = timeseries_data.rename(columns = changed_feats)
    
    return timeseries_data, changed_feats

def get_df_quantiles(df,
                     feats2check = timeseries_feats_columns,
                     feats2subdivide = ['orientation_food_edge'],
                     events2subdivide = ['food_region'],
                     feats2norm = feats2normalize,
                     is_remove_subdivided = True,
                     is_abs_ventral = True,
                     is_normalize = False
                     ):
    '''
    Get quantile statistics for all the features given by `feats2check`.
    In the features in `feats2abs` we are going to use only the absolute. This is to 
    deal with worms with unknown dorsal/ventral orientation.
    '''
    
    
    #filter default columns in case they are not present
    feats2check = [x for x in feats2check if x in df]
    feats2subdivide = [x for x in feats2check if x in feats2subdivide]
    events2subdivide = [x for x in df if x in events2subdivide]
    
    #subdivide a feature using the event features 
    if feats2subdivide and events2subdivide:
        subdivided_df = _get_subdivided_features(df, 
                             timeseries_cols = feats2subdivide, 
                             event_cols = events2subdivide)
        df = df.join(subdivided_df)
        feats2check += subdivided_df.columns.tolist()
        
        if is_remove_subdivided:
            df = df[[x for x  in df if not x in feats2subdivide]]
            feats2check = [x for x in feats2check if x not in feats2subdivide]
        
        
    if is_normalize:
        df, changed_feats = _normalize_by_w_length(df, feats2norm = feats2norm)
        feats2check = [x if not x in changed_feats else changed_feats[x] for x in feats2check]
        
    q_vals = [0.1, 0.5, 0.9]
    iqr_limits = [0.25, 0.75]
    
    valid_q = q_vals + iqr_limits
    
    feat_mean = None
    if is_abs_ventral:
        #find features that match ventral_signed_columns
        feats2abs = _h_filter_ventral_features(feats2check)
        if feats2abs:
            Q = df[feats2abs].abs().quantile(valid_q)
            Q.columns = [x+'_abs' for x in Q.columns]
            feats2check = [x for x in feats2check if not x in feats2abs]
            feat_mean = pd.concat((feat_mean, Q), axis=1)
    
    
    Q = df[feats2check].quantile(valid_q)
    feat_mean = pd.concat((feat_mean, Q), axis=1)
    
    
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


def _get_subdivided_features(timeseries_data, timeseries_cols, event_cols):
    '''
    event_vector = [-1, -1, 0, 0, 1, 1]
    feature_vector = [1, 3, 4, 5, 6, 6]
    
    new_vectors ->
    [1, 3, nan, nan, nan, nan]
    [nan, nan, 4, 5, nan, nan]
    [nan, nan, nan, nan, 6, 6]
    
    '''
    
    # I might need to input a more clever data for the subdivision like:
    #    [(feat, [event1, event2]), (feat2, (event1)), (feat3, (event2))
    
    assert all(x in event_region_labels.keys() for x in event_cols)
    
    event_type_link = {
            'food_region' : '_in_'
            }
    
    subdivided_data = []
    for e_col in event_cols:
        e_data = timeseries_data[e_col].values
        
        if e_col in event_type_link:
            str_l = event_type_link[e_col]
        else:
            str_l = '_'
        
        for flag, label in event_region_labels[e_col].items():
            _flag = e_data != flag
            
            for f_col in timeseries_cols:
                f_data = timeseries_data[f_col].values.copy()
                f_data[_flag] = np.nan
                
                new_name = f_col + str_l + label
                
                subdivided_data.append((new_name, f_data))
    
    columns, data = zip(*subdivided_data)
    subdivided_df = pd.DataFrame(np.array(data).T, columns = columns)
    subdivided_df.index = timeseries_data.index
    
    return subdivided_df
    

    #%%

def get_feat_stats(timeseries_data, fps, is_normalize):
    '''
    Get the features statistics from the features_timeseries, from both the
    quantiles and event data.
    '''
    
    n_worms_estimate = get_n_worms_estimate(timeseries_data['timestamp'])
    
    timeseries_stats_s = get_df_quantiles(timeseries_data, is_normalize = is_normalize)
    event_stats_s = get_event_stats(timeseries_data, fps , n_worms_estimate)
    path_grid_stats_s = get_path_extent_stats(timeseries_data, fps, is_normalized = is_normalize)
    
    #feat_stats_s['n_worms_estimate'] = n_worms_estimate
    
    feat_stats_s = pd.concat((timeseries_stats_s, event_stats_s, path_grid_stats_s))
    return feat_stats_s
    
def get_time_groups(timestamp, time_ranges_m, fps):
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



def get_feat_stats_binned(df, fps, time_ranges_m, is_normalize):
    '''
    Calculate the stats of the features in df using `get_feat_stats` by binning 
    the data according to its time. See `_get_time_groups` to see the kind of 
    values that `time_range_m` accepts.
    '''
    df['time_group'] = get_time_groups(df['timestamp'], time_ranges_m, fps)
    dat_agg = []
    for ind_t, dat in df.groupby('time_group'):
        if ind_t< 0:
            continue
        
        feat_means_s = get_feat_stats(dat, fps, is_normalize) 
        
        #add the time information at the begining
        feat_means_s = pd.Series(ind_t, index=['time_group']).append(feat_means_s)
        
        dat_agg.append(feat_means_s)
    
    dat_agg = pd.concat(dat_agg, axis=1).T
    return dat_agg

def collect_feat_stats(fnames, info_df, time_ranges_m = None, is_normalize = False):
    reserved_w = ['time_group', 'worm_index', 'timestamp']
    assert not any(x in info_df for x in reserved_w)
    
    all_data = []
    for (iexp, info_row), fname in zip(info_df.iterrows(), fnames):
        print(iexp + 1, len(fnames))
        
        with pd.HDFStore(fname, 'r') as fid:
            fps = fid.get_storer('/trajectories_data').attrs['fps']
            features_timeseries = fid['/timeseries_data']
            
        feat_mean_df = get_feat_stats_binned(features_timeseries, 
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

def _old_events_from_df(features_timeseries, fps):
    '''
    Calculate the event features, and its durations from the features_timeseries table
    '''
    from tierpsy_features.events import get_events
    
    event_list = []
    for worm_ind, worm_data in features_timeseries.groupby('worm_index'):
        df, durations_df = get_events(worm_data, fps)
        
        df.index = worm_data.index
        #insert worm index in the first position
        durations_df.insert(0, 'worm_index', worm_ind)
        event_list.append((df, durations_df))
    
    event_df, event_durations = zip(*event_list)
    event_df = pd.concat(event_df)
    
    event_durations = pd.concat(event_durations, ignore_index=True)
    return event_df, event_durations

def get_feat_stats_all(timeseries_data, blob_features, fps):
    
    feat_stats = get_feat_stats(timeseries_data, fps, is_normalize = False)
    
    
    #this is a dirty solution to avoid duplicates but we are testing right now
    feat_stats_n = get_feat_stats(timeseries_data, fps, is_normalize = True)
    feat_stats_n = feat_stats_n[[x for x in feat_stats_n.index if x not in feat_stats.index]]
    
    feat_stats_v = get_df_quantiles(timeseries_data, 
                     feats2check = ventral_signed_columns, 
                     is_abs_ventral = False, 
                     is_normalize = False)
    
    
    feat_stats_v_n = get_df_quantiles(timeseries_data, 
                     feats2check = ventral_signed_columns, 
                     is_abs_ventral = False, 
                     is_normalize = True)
    feat_stats_v_n = feat_stats_v_n[[x for x in feat_stats_v_n.index if x not in feat_stats_v.index]]
    
    blob_feats = [
           'area', 'perimeter', 'box_length', 'box_width',
           'quirkiness', 'compactness', 'solidity',
           'hu0', 'hu1', 'hu2', 'hu3', 'hu4',
           'hu5', 'hu6'
           ]
    blob_stats = get_df_quantiles(blob_features, feats2check = blob_feats)
    blob_stats.index = ['blob_' + x for x in blob_stats.index]
    
    exp_feats = pd.concat((feat_stats, feat_stats_n, blob_stats, feat_stats_v, feat_stats_v_n))
    
    return exp_feats

if __name__ == '__main__':
    import glob
    import sys
    import os
    
    from tierpsy.helper.params import read_ventral_side
    
    sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
    from misc import get_rig_experiments_df
    exp_set_dir = '/Users/ajaver/OneDrive - Imperial College London/swiss_strains'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
    
    set_type = 'featuresN'
    
    save_dir = './results_{}'.format(set_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments = get_rig_experiments_df(features_files, csv_files)
    
    info_cols = ['strain']
    
    fnames = [os.path.join(row['directory'], row['base_name'] + f_ext)
                for _, row in experiments.iterrows()]
    info_df = experiments[info_cols]
    
    assert info_df.shape[0] == len(fnames)
    
    all_data = []
    
    
    for fname, (ii, row) in zip(fnames, info_df.iterrows()):
        print(ii+1, len(info_df))
        
        ventral_side = read_ventral_side(fname)       
#        with pd.HDFStore(fname, 'r') as fid:
#            fps = fid.get_storer('/trajectories_data').attrs['fps']
#            timeseries_data = fid['/timeseries_data']
#            blob_features = fid['/blob_features']
#        
#        exp_feats = get_feat_stats_all(timeseries_data, blob_features, fps)
#        
#        info_zip = zip(*[row.tolist()]*len(exp_feats))
#        all_data += list(zip(*info_zip, exp_feats, exp_feats.index))
#        
#        print(len(all_data))
#        
#        break