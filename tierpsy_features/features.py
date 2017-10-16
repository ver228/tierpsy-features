#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:01:03 2017

@author: ajaver
"""
import pandas as pd
import numpy as np
import warnings

from tierpsy_features.velocities import get_velocity_features, velocities_columns
from tierpsy_features.postures import get_morphology_features, morphology_columns, \
get_posture_features, posture_columns, posture_aux

from tierpsy_features.curvatures import get_curvature_features, curvature_columns
from tierpsy_features.food import get_cnt_feats, food_columns
from tierpsy_features.path import get_path_curvatures, path_curvature_columns, path_curvature_columns_aux
from tierpsy_features.events import get_events, event_columns

timeseries_feats_columns = velocities_columns + morphology_columns + posture_columns + \
                curvature_columns + food_columns + path_curvature_columns

aux_columns =  posture_aux + path_curvature_columns_aux

timeseries_columns = timeseries_feats_columns + event_columns + aux_columns

ventral_signed_columns = [
        'angular_velocity',
        'relative_speed_midbody',
        'relative_angular_velocity_head_tip',
        'relative_angular_velocity_neck',
        'relative_angular_velocity_hips', 
        'relative_angular_velocity_tail_tip', 
        'eigen_projection_1', 
        'eigen_projection_2', 
        'eigen_projection_3', 
        'eigen_projection_4', 
        'eigen_projection_5', 
        'eigen_projection_6', 
        'eigen_projection_7', 
        ] + path_curvature_columns + curvature_columns


#all the ventral_signed_columns must be in timeseries_columns
assert len(set(ventral_signed_columns) - set(timeseries_columns))  == 0 

valid_ventral_side = ['','clockwise','anticlockwise', 'unknown']

def get_timeseries_features(skeletons, 
                            widths, 
                            dorsal_contours, 
                            ventral_contours,
                            fps,
                            ventral_side = '',
                            timestamp = None,
                            food_cnt = None,
                            is_smooth_cnt = False,
                            delta_time = 1/3, #delta time in seconds to calculate the velocity
                            ):
    
    assert ventral_side in valid_ventral_side

    feat_morph = get_morphology_features(skeletons, widths, dorsal_contours, ventral_contours)
    feat_posture = get_posture_features(skeletons)
    
    #I am still missing the velocity and path features but it should look like this
    cols_to_use = [x for x in feat_posture.columns if x not in feat_morph] #avoid duplicate length
    
    features_df = feat_morph.join(feat_posture[cols_to_use])
    
    curvatures = get_curvature_features(skeletons)
    features_df = features_df.join(curvatures)
    
    velocities = get_velocity_features(skeletons, delta_time, fps)
    if velocities is not None:
        features_df = features_df.join(velocities)
    
    if food_cnt is not None:
        food = get_cnt_feats(skeletons, 
                             food_cnt,
                             is_smooth_cnt
                             )
        features_df = features_df.join(food)
    
    
    path_curvatures, path_coords = get_path_curvatures(skeletons)
    features_df = features_df.join(path_curvatures)
    features_df = features_df.join(path_coords)
    
    
    if timestamp is None:
        timestamp = np.arange(features_df.shape[0], np.int32)
        warnings.warn('`timestamp` was not given. I will assign an arbritary one.')
    
    features_df['timestamp'] = timestamp
    
    events_df = get_events(features_df, fps)
    
    dd = [x for x in events_df if x in event_columns]
    features_df = features_df.join(events_df[dd])
    
    #add any missing column
    all_columns = ['timestamp'] + timeseries_columns
    df = pd.DataFrame([], columns = timeseries_columns)
    features_df = pd.concat((df, features_df), ignore_index=True)
    features_df = features_df[all_columns]
    
    #correct ventral side sign
    if ventral_side == 'clockwise':
        features_df[ventral_signed_columns] *= -1
    
    return features_df