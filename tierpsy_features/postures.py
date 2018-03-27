#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:01:03 2017

@author: ajaver
"""
import numpy as np
import tables
import warnings
import cv2
import pandas as pd
from collections import OrderedDict

from tierpsy_features.helper import DataPartition
from tierpsy_features import EIGEN_PROJECTION_FILE

morphology_columns = ('length', 'area', 'width_head_base', 'width_midbody', 'width_tail_base')

posture_columns = ('quirkiness', 'major_axis',
       'minor_axis', 'eigen_projection_1', 'eigen_projection_2',
       'eigen_projection_3', 'eigen_projection_4', 'eigen_projection_5',
       'eigen_projection_6', 'eigen_projection_7')

posture_aux = ('head_tail_distance',)

#%% Morphology Features
def get_widths(widths):
    partitions = ('head_base', 'midbody', 'tail_base')
    p_obj = DataPartition(partitions, n_segments=widths.shape[1])
    
    with warnings.catch_warnings():
        #I am unwraping in one dimension first
        warnings.simplefilter("ignore")
        segment_widths = {p:p_obj.apply(widths, p, func=np.median) for p in partitions}

    return segment_widths

def _h_signed_areas(cnt_side1, cnt_side2):
    '''calculate the contour area using the shoelace method, the sign indicate the contour orientation.'''
    assert cnt_side1.shape == cnt_side2.shape
    if cnt_side1.ndim == 2:
        # if it is only two dimenssion (as if in a single skeleton).
        # Add an extra dimension to be compatible with the rest of the code
        cnt_side1 = cnt_side1[None, ...]
        cnt_side2 = cnt_side2[None, ...]

    contour = np.hstack((cnt_side1, cnt_side2[:, ::-1, :]))
    signed_area = np.sum(
        contour[:,:-1,0] * contour[:,1:,1] -
        contour[:,1:,0] * contour[:,:-1,1],
        axis=1)/ 2
    
    assert signed_area.size == contour.shape[0]
    return signed_area

def get_area(cnt_side1, cnt_side2):
    with np.errstate(invalid='ignore'):
        area = np.abs(_h_signed_areas(cnt_side1, cnt_side2))
    return area

def get_length(skeletons):
    '''
    Calculate length using the skeletons
    '''
    delta_coords = np.diff(skeletons, axis=1)
    segment_sizes = np.linalg.norm(delta_coords, axis=2)
    w_length = np.sum(segment_sizes, axis=1)
    return w_length


def get_morphology_features(skeletons, 
                            widths = None, 
                            dorsal_contours = None, 
                            ventral_contours = None):
    
    data = OrderedDict()
    
    lengths = get_length(skeletons)
    data['length'] = lengths
    
    areas = None
    if ventral_contours is not None and dorsal_contours is not None:
        areas = get_area(ventral_contours, dorsal_contours)
        data['area'] = areas
        #data['area_length_ratio'] = areas/lengths
    
    if widths is not None:
        widths_seg = get_widths(widths)
        #data['width_length_ratio'] = widths_seg['midbody']/lengths
        for p in widths_seg:
            data['width_' + p] = widths_seg[p]
       
    data = pd.DataFrame.from_dict(data)
    return data


#%%
def _h_angles(skeletons):
    dd = np.diff(skeletons,axis=1);
    angles = np.arctan2(dd[...,0], dd[...,1])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1);
    
    mean_angles = np.mean(angles, axis=1)
    angles -= mean_angles[:, None]
    
    return angles, mean_angles

def get_eigen_projections(skeletons):
    with tables.File(EIGEN_PROJECTION_FILE) as fid:
        eigen_worms = fid.get_node('/eigenWorms')[:]
        eigen_worms = eigen_worms.T
    
    angles, _ = _h_angles(skeletons)   
    eigen_projections = np.dot(eigen_worms, angles.T)
    eigen_projections = np.rollaxis(eigen_projections, -1, 0)
    return eigen_projections

#%%
def get_quirkiness(skeletons):
    bad = np.isnan(skeletons[:, 0, 0])
    
    dd = [cv2.minAreaRect(x) for x in skeletons.astype(np.float32)]
    dd = [(L,W) if L >W else (W,L) for _,(L,W),_ in dd]
    L, W = list(map(np.array, zip(*dd)))
    L[bad] = np.nan
    W[bad] = np.nan
    quirkiness = np.sqrt(1 - W**2 / L**2)
    
    return quirkiness, L, W

def get_head_tail_dist(skeletons):
    return np.linalg.norm(skeletons[:, 0, :] - skeletons[:, -1, :], axis=1)
#%%
def get_posture_features(skeletons):
    
    
    head_tail_dist = get_head_tail_dist(skeletons)
    quirkiness, major_axis, minor_axis = get_quirkiness(skeletons)
    
    #I prefer to explicity recalculate the lengths, just to do not have to pass the length information
    eigen_projections = get_eigen_projections(skeletons)
    
    #repack into an ordered dictionary
    data = OrderedDict(
        [
        ('head_tail_distance' , head_tail_dist),
        ('quirkiness' , quirkiness),
        ('major_axis' , major_axis),
        ('minor_axis' , minor_axis)
        ]
    )
    
    for n in range(eigen_projections.shape[1]):
        data['eigen_projection_' + str(n+1)] = eigen_projections[:, n]
    
    data = pd.DataFrame.from_dict(data)
    return data

#%%
if __name__ == '__main__':
    data = np.load('worm_example_small_W1.npz')
    
    skeletons = data['skeleton']
    dorsal_contours = data['dorsal_contour']
    ventral_contours = data['ventral_contour']
    widths = data['widths']
    
    feat_morph = get_morphology_features(skeletons, widths, dorsal_contours, ventral_contours)
    feat_posture = get_posture_features(skeletons, curvature_window = 4)
    
    #I am still missing the velocity and path features but it should look like this
    cols_to_use = [x for x in feat_posture.columns if x not in feat_morph] #avoid duplicate length
    
    features = feat_morph.join(feat_posture[cols_to_use])
    