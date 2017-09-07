#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:01:03 2017

@author: ajaver
"""
from .velocities import get_velocity_features
from .postures import get_morphology_features, get_posture_features
from .curvatures import get_curvature_features

def get_timeseries_features(skeletons, 
                            widths, 
                            dorsal_contours, 
                            ventral_contours,
                            fps,
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
    
    return features