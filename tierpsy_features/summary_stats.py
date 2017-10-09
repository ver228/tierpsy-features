#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:24:25 2017

@author: ajaver
"""

import pandas as pd
import numpy as np

from tierpsy_features.events import get_events
from tierpsy_features.features import timeseries_columns, ventral_signed_columns

def _normalize_by_w_length(features_timeseries):
    #%%
    features_timeseries = features_timeseries.copy()
    
    median_length = features_timeseries.groupby('worm_index').agg({'length':'median'})
    median_length_vec = features_timeseries['worm_index'].map(median_length['length'])
    
    feats2norm = [
           'speed',
           'relative_speed_midbody', 
           'relative_radial_velocity_head_tip',
           'relative_radial_velocity_neck',
           'relative_radial_velocity_hips',
           'relative_radial_velocity_tail_tip',
           'head_tail_distance',
           'major_axis', 
           'minor_axis', 
           'dist_from_food_edge'
           ]

    for f in feats2norm:
        features_timeseries[f] /= median_length_vec
    
    curv_feats = ['curvature_head',
                   'curvature_hips', 
                   'curvature_midbody', 
                   'curvature_neck',
                   'curvature_tail']
    
    
    for f in curv_feats:
        features_timeseries[f] *= median_length_vec
        
    changed_feats = {x: x + '_norm' for x in feats2norm + curv_feats}    
    features_timeseries = features_timeseries.rename(columns = changed_feats)
    
    return features_timeseries, changed_feats

def get_n_worms_estimate(frame_numbers, percentile = 99):
    '''
    Get an estimate of the number of worms using the table frame_numbers vector
    '''
    
    n_per_frame = frame_numbers.value_counts()
    n_per_frame = n_per_frame.values
    if len(n_per_frame) > 0:
        n_worms_estimate = np.percentile(n_per_frame, percentile)
    else:
        n_worms_estimate = 0
    return n_worms_estimate


def get_df_quantiles(df, 
                     ventral_side = None, 
                     feats2abs = ventral_signed_columns,
                     feats2check = timeseries_columns,
                     is_normalize = False
                     ):
    '''
    Get quantile statistics for all the features given by `feats2check`.
    In the features in `feats2abs` we are going to use only the absolute. This is to 
    deal with worms with unknown dorsal/ventral orientation.
    '''
    
    if is_normalize:
        df, changed_feats = _normalize_by_w_length(df)
        feats2abs = [x if not x in changed_feats else changed_feats[x] for x in feats2abs]
        feats2check = [x if not x in changed_feats else changed_feats[x] for x in feats2check]
        
    
    
    q_vals = [0.1, 0.5, 0.9]
    iqr_limits = [0.25, 0.75]
    
    valid_q = q_vals + iqr_limits
    
    if not ventral_side in ['clockwise', 'anticlockwise']:
        
        
        
        Q = df[feats2abs].abs().quantile(valid_q)
        Q.columns = [x+'_abs' for x in Q.columns]
    
    vv = [x for x in feats2check if not x in feats2abs]
    Q_s = df[vv].quantile(valid_q)
    feat_mean = pd.concat((Q, Q_s), axis=1)
    
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

def events_from_df(features_timeseries, fps):
    '''
    Calculate the event features, and its durations from the features_timeseries table
    '''
    
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
        


def get_event_stats(event_durations, n_worms_estimate, total_time):
    '''
    Get the event statistics using the event durations table.
    '''
    
    region_labels = {
            'motion_mode': {-1:'backward', 1:'forward', 0:'paused'}, 
            'food_region': {-1:'outside', 1:'inside', 0:'edge'}
            }
    
    tot_times = event_durations.groupby('event_type').agg({'duration':'sum'})['duration']
    event_g = event_durations.groupby(('event_type', 'region'))
    event_stats = []
    for event_type, region_dict in region_labels.items():
        for region_id, region_name in region_dict.items():
            stat_prefix = event_type + '_' + region_name
            try:
                dat = event_g.get_group((event_type, region_id))
                duration = dat['duration'].values
                edge_flag = dat['edge_flag'].values
            except:
                duration = np.zeros(1)
                edge_flag = np.zeros(0)
    
            stat_name = stat_prefix + '_duration_50th'
            stat_val = np.nanmedian(duration)
            event_stats.append((stat_val, stat_name))
            
            stat_name = stat_prefix + '_fraction'
            stat_val = np.nansum(duration)/tot_times[event_type]
            event_stats.append((stat_val, stat_name))
            
            stat_name = stat_prefix + '_frequency'
            # calculate total events excluding events that started before the beginig of the trajectory
            total_events = (edge_flag != -1).sum()
            stat_val = total_events/n_worms_estimate/total_time
            event_stats.append((stat_val, stat_name))
            
    event_stats_s = pd.Series(*list(zip(*event_stats)))
    return event_stats_s

def get_feat_stats(features_timeseries, fps, is_normalize):
    '''
    Get the features statistics from the features_timeseries, from both the
    quantiles and event data.
    '''
    
    feat_mean_s = get_df_quantiles(features_timeseries, is_normalize = is_normalize)
    
    event_df, event_durations = events_from_df(features_timeseries.copy(), fps)
    
    n_worms_estimate = get_n_worms_estimate(features_timeseries['timestamp'])
    total_time = (features_timeseries['timestamp'].max() - features_timeseries['timestamp'].min())/fps
    
    event_stats_s = get_event_stats(event_durations, n_worms_estimate, total_time)
    feat_mean_s = feat_mean_s.append(event_stats_s)
    
    #%%
    return feat_mean_s
    
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

#def collect_feat_stats(fnames, info_df):
#    assert info_df.shape[0] == len(fnames)
#    exp_ids = info_df.index
#    all_data = []
#    for iexp, fname in zip(exp_ids, fnames):
#        print(iexp + 1, len(fnames))
#        
#        with pd.HDFStore(fname, 'r') as fid:
#            fps = fid.get_storer('/trajectories_data').attrs['fps']
#            features_timeseries = fid['/timeseries_features']
#        feat_mean_s = get_feat_stats(features_timeseries, fps)
#        
#        all_data.append((iexp, feat_mean_s))
#    
#    exp_inds, feat_means = zip(*all_data)
#    feat_means_df = pd.concat(feat_means, axis=1).T
#    feat_means_df.index = exp_inds
#    feat_means_df = info_df.join(feat_means_df)
#    
#    return feat_means_df

def collect_feat_stats(fnames, info_df, time_ranges_m = None, is_normalize = False):
    reserved_w = ['time_group', 'worm_index', 'timestamp']
    assert not any(x in info_df for x in reserved_w)
    
    all_data = []
    for (iexp, info_row), fname in zip(info_df.iterrows(), fnames):
        print(iexp + 1, len(fnames))
        
        with pd.HDFStore(fname, 'r') as fid:
            fps = fid.get_storer('/trajectories_data').attrs['fps']
            features_timeseries = fid['/timeseries_features']
            
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
    
    if collect_feat_stats is None:
        #this feature should be all zeros and it is confusing if there was no binning done
        del feat_means_df['time_group'] 
    
    return feat_means_df

if __name__ == '__main__':
    import glob
    import sys
    import os
    sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
    from misc import get_rig_experiments_df
    
    
    #exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Swiss_Strains'
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
    
    fname = fnames[0]
    
    with pd.HDFStore(fname, 'r') as fid:
        fps = fid.get_storer('/trajectories_data').attrs['fps']
        features_timeseries = fid['/timeseries_features']
        feat_mean_s = get_df_quantiles(features_timeseries, is_normalize = True)
#    #%%
#    event_vector, event_durations = events_from_df(features_timeseries, fps)
#    #%%
#    
#    feat_means_df = collect_feat_stats(fnames, info_df)
#    feat_means_df.to_csv('swiss_strains_stats.csv', index=False)
    
    
    
    