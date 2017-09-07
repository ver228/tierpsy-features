#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:54:05 2017

@author: ajaver
"""

from tierpsy_features import get_timeseries_features
import numpy as np

if __name__ == '__main__':
    #data = np.load('../notebooks/data/worm_example.npz')
    #data = np.load('../notebooks/data/worm_example_small_W1.npz')
    data = np.load('../notebooks/data/worm_example_big_W1.npz')
    
    skeletons = data['skeleton']
    dorsal_contours = data['dorsal_contour']
    ventral_contours = data['ventral_contour']
    widths = data['widths']
    
    timeseries_features = get_timeseries_features(
            skeletons, 
            widths, 
            dorsal_contours, 
            ventral_contours,
            fps = 25
            )

    
    #%%
    corr_mat = timeseries_features.corr()
    #corr_mat = features.corr('spearman')
    gg = [corr_mat.index[corr_mat[col].abs()>0.9] for col in corr_mat]
    gg = [list(x) for x in gg if len(x) > 1]
    
    gg_reduced = []
    for x in gg:
        if not x in gg_reduced:
            gg_reduced.append(x)
    
    
    gg_reduced = sorted(gg_reduced, key = lambda x : np.abs(corr_mat.loc[x[0], x[1]]))[::-1]
    for x in gg_reduced:
        print(corr_mat.loc[x[0], x[1]], x)
        

##%%
#'''
#0.967591708477 ['head_tail_distance', 'major_axis']
#0.945765029426 ['area', 'area_length_ratio']
#0.938536562601 ['curvature_midbody', 'eigen_projection_3']
#'''
#
#'''
#0.994695366285 ['head_tail_distance', 'major_axis', 'minor_axis']
#0.994695366285 ['head_tail_distance', 'major_axis']
#0.942735018206 ['area', 'area_length_ratio']
#0.906450160785 ['curvature_midbody', 'eigen_projection_3']
#-0.905338356343 ['major_axis', 'minor_axis']
#'''
# 
#'''
#0.9860452676 ['head_tail_distance', 'major_axis']
#'''
#    #%%
##SPEARMAN
#    
#'''
#0.996958562074 ['head_tail_distance', 'major_axis']
#-0.983479760453 ['quirkiness', 'minor_axis']
#0.941203696832 ['curvature_midbody', 'eigen_projection_3']
#0.908940424064 ['area', 'area_length_ratio']
#0.900330397693 ['quirkiness', 'major_axis', 'minor_axis']
#0.890882931537 ['head_tail_distance', 'quirkiness', 'major_axis']
#'''
#
#'''
#0.999023464384 ['head_tail_distance', 'major_axis']
#-0.994472149972 ['quirkiness', 'minor_axis']
#0.939077002241 ['area', 'area_length_ratio']
#0.918544010619 ['curvature_midbody', 'eigen_projection_3']
#'''
#
#'''
#0.998847729215 ['head_tail_distance', 'major_axis']
#-0.987756561859 ['quirkiness', 'minor_axis']
#'''
#
##%%
##might be useful to find pre-coiling shapes
#dd = features['head_tail_distance']-features['major_axis']

#%%
