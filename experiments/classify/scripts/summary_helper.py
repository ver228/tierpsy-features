#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:04:47 2018

@author: ajaver
"""

import numpy as np
import pandas as pd
import tables

from tierpsy_features import get_timeseries_features, timeseries_feats_columns, ventral_signed_columns
from tierpsy_features.summary_stats import blob_feats_columns, get_event_stats, \
 get_df_quantiles, get_n_worms_estimate, get_path_extent_stats, feats2normalize

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name, TABLE_FILTERS
from tierpsy.helper.params import read_fps, read_ventral_side

from functools import partial
from summary_CeNDR import read_ow_feats

from tierpsy_features.summary_stats import _test_get_feat_stats_all
from tierpsy_features.velocities import get_velocity, get_relative_velocities
import math



feats2normalize['L'] = feats2normalize['L'] + ['relative_neck_radial_velocity_head_tip', 'relative_hips_radial_velocity_tail_tip']

ventral_signed_columns = ventral_signed_columns +  ['relative_neck_angular_velocity_head_tip',
                             'relative_hips_angular_velocity_tail_tip'
                             ] 
timeseries_feats_columns = timeseries_feats_columns + ['relative_neck_angular_velocity_head_tip',
                             'relative_neck_radial_velocity_head_tip',
                             'relative_hips_angular_velocity_tail_tip',
                             'relative_hips_radial_velocity_tail_tip'
                             ] 

def _get_tips_relative_speed(skeletons, delta_frames, fps):
    #%%
    _, _, skel_centre_neck = get_velocity(skeletons, 'neck', delta_frames, fps)
    r_rad_to_neck, r_angular_to_neck = get_relative_velocities(skel_centre_neck, ['head_tip'], delta_frames, fps)

    signed_speed_hips, angular_velocity_hips, skel_centre_hips = get_velocity(skeletons, 'hips', delta_frames, fps)
    r_rad_to_hips, r_angular_to_hips = get_relative_velocities(skel_centre_hips, ['tail_tip'], delta_frames, fps)
    #%%
    feats = {
            'relative_neck_angular_velocity_head_tip' : r_angular_to_neck['head_tip'],
            'relative_neck_radial_velocity_head_tip' : r_rad_to_neck['head_tip'],
            'relative_hips_angular_velocity_tail_tip' : r_angular_to_hips['tail_tip'],
            'relative_hips_radial_velocity_tail_tip' : r_rad_to_hips['tail_tip']
            }
    
    return feats

#%%
def _add_derivatives(feats, cols2deriv, delta_frames, delta_time):
    
    assert feats['worm_index'].unique().size == 1
    feats = feats.sort_values(by='timestamp')
    
    df_ts = feats[cols2deriv].copy()
    df_ts.columns = ['d_' + x for x in df_ts.columns]
    
    m_o, m_f = math.floor(delta_frames/2), math.ceil(delta_frames/2)
    
    
    vf = df_ts.iloc[delta_frames:].values
    vo = df_ts.iloc[:-delta_frames].values
    vv = (vf - vo)/delta_time
    
    #the series was too small to calculate the derivative
    if vv.size > 0:
        df_ts.loc[:] =  np.nan
        df_ts.iloc[m_o:-m_f] = vv
        
    feats = pd.concat([feats, df_ts], axis=1)
    
    return feats
#%%
def _get_timeseries_feats(features_file, delta_time = 1/3):
    #%%
    timeseries_features = []
    fps = read_fps(features_file)
    
    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    #only use data that was skeletonized
    #trajectories_data = trajectories_data[trajectories_data['skeleton_id']>=0]
    
    trajectories_data_g = trajectories_data.groupby('worm_index_joined')
    progress_timer = TimeCounter('')
    base_name = get_base_name(features_file)
    tot_worms = len(trajectories_data_g)
    
    def _display_progress(n):
            # display progress
        dd = " Calculating tierpsy features. Worm %i of %i done." % (n+1, tot_worms)
        print_flush(
            base_name +
            dd +
            ' Total time:' +
            progress_timer.get_time_str())
    
    _display_progress(0)
    
    with tables.File(features_file, 'r') as fid:
        if '/food_cnt_coord' in fid:
            food_cnt = fid.get_node('/food_cnt_coord')[:]
        else:
            food_cnt = None
    
        #If i find the ventral side in the multiworm case this has to change
        ventral_side = read_ventral_side(features_file)
            
        timeseries_features = []
        for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
            with tables.File(features_file, 'r') as fid:
                skel_id = worm_data['skeleton_id'].values
                
                #deal with any nan in the skeletons
                good_id = skel_id>=0
                skel_id_val = skel_id[good_id]
                traj_size = skel_id.size

                args = []
                for p in ('skeletons', 'widths', 'dorsal_contours', 'ventral_contours'):
                    node = fid.get_node('/coordinates/' + p)
                    
                    dat = np.full((traj_size, *node.shape[1:]), np.nan)
                    if skel_id_val.size > 0:
                        if len(node.shape) == 3:
                            dd = node[skel_id_val, :, :]
                        else:
                            dd = node[skel_id_val, :]
                        dat[good_id] = dd
                    
                    args.append(dat)

                timestamp = worm_data['timestamp_raw'].values.astype(np.int32)
            
            feats = get_timeseries_features(*args, 
                                           timestamp = timestamp,
                                           food_cnt = food_cnt,
                                           fps = fps,
                                           ventral_side = ventral_side
                                           )
            #save timeseries features data
            feats = feats.astype(np.float32)
            feats['worm_index'] = worm_index
            #move the last fields to the first columns
            cols = feats.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            feats = feats[cols]
            
            feats['worm_index'] = feats['worm_index'].astype(np.int32)
            feats['timestamp'] = feats['timestamp'].astype(np.int32)
            
            #% Tests
            #append features relative to the neck and hips. I am not sure if i will included them as default
            delta_frames = int(round(fps*delta_time))
            skeletons = args[0]
            ff = _get_tips_relative_speed(skeletons, delta_frames, fps)
            for cc in ff:
                feats[cc] = ff[cc]
                #correct ventral side sign
                if ventral_side == 'clockwise' and 'angular' in cc:
                    feats[cc] *= -1
            
            
            
            
            feats = _add_derivatives(feats, timeseries_feats_columns, delta_frames, delta_time)
            #%
            
            
            timeseries_features.append(feats)
            _display_progress(ind_n)
        
        timeseries_features = pd.concat(timeseries_features, ignore_index=True)
    #%%
    return timeseries_features

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

def _get_feat_stats(timeseries_data, blob_features, fps):
    # be careful the blob features has some repeated names like area...
    #%%
    ts_cols_all = timeseries_feats_columns + ['d_' + x for x in timeseries_feats_columns]
    v_sign_cols = ventral_signed_columns + ['d_' + x for x in ventral_signed_columns]
    
    blob_cols = ['blob_' + x for x in blob_feats_columns]
    blob_cols = blob_cols + ['d_' + x for x in blob_cols]
    
    
    feats2norm = {}
   
    for k,dat in feats2normalize.items():
        feats2norm[k] = dat + ['d_' + x for x in dat]
    ts_cols_norm = sum(feats2norm.values(), [])
    
    n_worms_estimate = get_n_worms_estimate(timeseries_data['timestamp'])
    event_stats_s = get_event_stats(timeseries_data, fps , n_worms_estimate)
    
    #normal
    timeseries_stats_s = get_df_quantiles(timeseries_data, 
                                          feats2check = ts_cols_all,
                                          feats2abs = v_sign_cols,
                                          is_normalize = False)
    
    path_grid_stats_s = get_path_extent_stats(timeseries_data, fps, is_normalized = False)
    
    feat_stats = pd.concat((timeseries_stats_s, path_grid_stats_s, event_stats_s))
    #%%
    #normalized by worm length
    timeseries_stats_n = get_df_quantiles(timeseries_data, 
                                          feats2check = ts_cols_norm,
                                          feats2abs = v_sign_cols,
                                          feats2norm = feats2norm, 
                                          is_normalize = True)
    
    path_grid_stats_n = get_path_extent_stats(timeseries_data, fps, is_normalized = True)
    feat_stats_n = pd.concat((timeseries_stats_n, path_grid_stats_n))
    #%%
    
    
    #non-abs ventral signed features
    feat_stats_v = get_df_quantiles(timeseries_data, 
                                          feats2check = v_sign_cols,
                                          feats2abs = v_sign_cols,
                                          is_abs_ventral = False,
                                          is_normalize = False)
    
    v_sign_cols_norm = list(set(v_sign_cols) & set(ts_cols_norm))
    #non-abs and normalized ventral signed features
    feat_stats_v_n = get_df_quantiles(timeseries_data, 
                                          feats2check = v_sign_cols_norm,
                                          feats2abs = v_sign_cols,
                                          feats2norm = feats2norm,
                                          is_abs_ventral = False,
                                          is_normalize = True)
    
    
    #feat_stats_v_n = feat_stats_v_n[[x for x in feat_stats_v_n.index if x not in feat_stats_v.index]]
    #%%
    blob_stats = get_df_quantiles(blob_features, feats2check = blob_cols)
    #blob_stats.index = ['blob_' + x for x in blob_stats.index]
    
    
    blob_features['motion_mode'] = timeseries_data['motion_mode']
    blob_stats_m_subdiv = get_df_quantiles(blob_features, 
                                      feats2check = blob_cols, 
                                      subdivision_dict = {'motion_mode':blob_cols}, 
                                      is_abs_ventral = False)
    #blob_stats_m_subdiv.index = ['blob_' + x for x in blob_stats_m_subdiv.index]
    
    feat_stats_m_subdiv = get_df_quantiles(timeseries_data, 
                                      feats2check = ts_cols_all, 
                                      subdivision_dict = {'motion_mode':ts_cols_all}, 
                                      is_abs_ventral = False)
    #%%
    
    exp_feats = (feat_stats, 
                           feat_stats_m_subdiv, 
                           feat_stats_n, 
                           blob_stats, 
                           blob_stats_m_subdiv, 
                           feat_stats_v, 
                           feat_stats_v_n)
    
    exp_feats = pd.concat(exp_feats)
    return exp_feats

def process_feat_tierpsy_file(fname, delta_time = 1/3):
    #%%
    fps = read_fps(fname)
    delta_frames = int(round(fps*delta_time))
    timeseries_data = _get_timeseries_feats(fname, delta_time)
    #%%
    with pd.HDFStore(fname, 'r') as fid:
        blob_features = fid['/blob_features']
        
    blob_features = blob_features[blob_feats_columns]
    blob_features.columns = ['blob_' + x for x in blob_feats_columns]
    
    b_feats = blob_features.columns
    index_cols = ['worm_index', 'timestamp']
    
    blob_features = pd.concat((timeseries_data[index_cols], blob_features), axis=1)
    #%%
    #add derivatives
    blob_l = []
    for w_ind, blob_w in blob_features.groupby('worm_index'):
        blob_w = _add_derivatives(blob_w, b_feats, delta_frames, delta_time)
        blob_l.append(blob_w)
    blob_features = pd.concat(blob_l, axis=0)
    #%%
    exp_feats = _get_feat_stats(timeseries_data, blob_features, fps)
    #%%
    return exp_feats



if __name__ == '__main__':
    #fname =  '/Volumes/behavgenom_archive$/single_worm/finished/WT/AQ2947/food_OP50/XX/30m_wait/anticlockwise/483 AQ2947 on food R_2012_03_08__15_42_48___1___8_featuresN.hdf5'
    #fname = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/del-1(ok150)X@NC279/food_OP50/XX/30m_wait/clockwise/del-1 (ok150)X on food L_2012_03_08__15_16_22___1___7_featuresN.hdf5'
    fname = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/gpa-6(pk480)X@NL1146/food_OP50/XX/30m_wait/clockwise/gpa-6 (ph480)X on food L_2009_07_16__12_40__3_featuresN.hdf5'
    #fname = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_020617/N2_worms5_food1-10_Set1_Pos4_Ch5_02062017_115615_featuresN.hdf5'
    
    
    delta_time = 1/3
    exp_feats = process_feat_tierpsy_file(fname, delta_time)
    
    #make sure all the features are unique
    assert np.unique(exp_feats.index).size == exp_feats.size
    #%%
    #for x in exp_feats.index:
    #    print(x)
    
    