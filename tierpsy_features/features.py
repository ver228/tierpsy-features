#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:01:03 2017

@author: ajaver
"""
import pandas as pd

from tierpsy_features.velocities import get_velocity_features, velocities_columns
from tierpsy_features.postures import get_morphology_features, morphology_columns, \
get_posture_features, posture_columns

from tierpsy_features.curvatures import get_curvature_features, curvature_columns
from tierpsy_features.food import get_cnt_feats, food_columns

timeseries_columns = velocities_columns + morphology_columns + posture_columns + \
                curvature_columns + food_columns

ventral_signed_columns = [
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
    'curvature_head', 
    'curvature_hips', 
    'curvature_midbody', 
    'curvature_neck', 
    'curvature_tail'
    ]

#all the ventral_signed_columns must be in timeseries_columns
assert len(set(ventral_signed_columns) - set(timeseries_columns))  == 0 

#%%
def get_timeseries_features(skeletons, 
                            widths, 
                            dorsal_contours, 
                            ventral_contours,
                            fps,
                            food_cnt = None,
                            is_smooth_cnt = False,
                            delta_time = 1/3, #delta time in seconds to calculate the velocity
                            curvature_window = 1
                            ):
    
    feat_morph = get_morphology_features(skeletons, widths, dorsal_contours, ventral_contours)
    feat_posture = get_posture_features(skeletons)
    
    #I am still missing the velocity and path features but it should look like this
    cols_to_use = [x for x in feat_posture.columns if x not in feat_morph] #avoid duplicate length
    
    features = feat_morph.join(feat_posture[cols_to_use])
    
    curvatures = get_curvature_features(skeletons, points_window=curvature_window)
    features = features.join(curvatures)
    
    velocities = get_velocity_features(skeletons, delta_time, fps)
    if velocities is not None:
        features = features.join(velocities)
    
    if food_cnt is not None:
        food = get_cnt_feats(skeletons, 
                             food_cnt,
                             is_smooth_cnt
                             )
        features = features.join(food)
    
    #add any missing column
    df = pd.DataFrame([], columns=timeseries_columns)
    features = pd.concat((df, features), ignore_index=True)
    features = features[timeseries_columns]
    return features