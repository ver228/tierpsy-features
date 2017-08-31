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

from tierpsy_features.helper import DataPartition, nanunwrap
from tierpsy_features import EIGEN_PROJECTION_FILE

#%% Morphology Features
def get_widths(widths):
    partitions = ['head_base', 'midbody', 'tail_base']
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
    return np.abs(_h_signed_areas(cnt_side1, cnt_side2))

def get_length(skeletons):
    '''
    Calculate length using the skeletons
    '''
    delta_coords = np.diff(skeletons, axis=1)
    segment_sizes = np.linalg.norm(delta_coords, axis=2)
    w_length = np.sum(segment_sizes, axis=1)
    return w_length


def get_morphology_features(skeletons, widths, dorsal_contours, ventral_contours):
    widths_seg = get_widths(widths)
    areas = get_area(ventral_contours, dorsal_contours)
    lengths = get_length(skeletons)
    
    data = OrderedDict(
        [
        ('length', lengths),
        ('area' , areas),
        ('area_length_ratio' , areas/lengths),
        ('width_length_ratio' , areas/widths_seg['midbody'])
        ]
    )
    for p in widths_seg:
        data['width_' + p] = widths_seg[p]
    data = pd.DataFrame.from_dict(data)
    return data

#%% properties
def _h_tangent_angles(skels, points_window):
    '''this is a vectorize version to calculate the angles between segments
    segment_size points from each side of a center point.
    '''
    s_center = skels[:, points_window:-points_window, :] #center points
    s_left = skels[:, :-2*points_window, :] #left side points
    s_right = skels[:, 2*points_window:, :] #right side points
    
    d_left = s_left - s_center 
    d_right = s_center - s_right
    
    #arctan2 expects the y,x angle
    ang_l = np.arctan2(d_left[...,1], d_left[...,0])
    ang_r = np.arctan2(d_right[...,1], d_right[...,0])
    
    with warnings.catch_warnings():
        #I am unwraping in one dimension first
        warnings.simplefilter("ignore")
        ang = np.unwrap(ang_r-ang_l, axis=1);
    
    for ii in range(ang.shape[1]):
        ang[:, ii] = nanunwrap(ang[:, ii])
    return ang

def _h_curvature(skeletons, points_window, lengths=None):
    if lengths is None:
        #caculate the length if it is not given
        lengths = get_length(skeletons)
    
    #Number of segments is the number of vertices minus 1
    n_segments = skeletons.shape[1] -1 
    
    #This is the fraction of the length the angle is calculated on
    length_frac = 2*(points_window-1)/(n_segments-1)
    segment_length = length_frac*lengths
    segment_angles = _h_tangent_angles(skeletons, points_window)
    
    curvature = segment_angles/segment_length[:, None]
    
    return curvature
    
def get_curvature(skeletons, points_window, lengths=None):
    segments_ind_dflt = {
        'head' : 0,
        'neck' : 0.25,
        'midbody' : 0.5, 
        'hips' : 0.75,
        'tail' : 1.,
    }
    
    curvatures = _h_curvature(skeletons, points_window, lengths)
    max_angle_index = curvatures.shape[-1]-1
    segments_ind = {k:int(round(x*max_angle_index)) for k,x in segments_ind_dflt.items()}
    
    curv_dict = {x:curvatures[:, ind] for x,ind in segments_ind.items()}
    return curv_dict


def _h_curvature_test(skeletons):
    '''
    Calculate the curvature using univariate splines. This method is slower and can fail
    badly if the fit does not work, so I am only using it as testing
    '''
    from scipy.interpolate import UnivariateSpline
    
    def _get_curvature(skel):
        if np.any(np.isnan(skel)):
            return np.full(skel.shape[0], np.nan)
        
        x = skel[:, 0]
        y = skel[:, 1]
        n = np.arange(x.size)
    
        fx = UnivariateSpline(n, x, k=5)
        fy = UnivariateSpline(n, y, k=5)
    
        x_d = fx.derivative(1)(n)
        x_dd = fx.derivative(2)(n)
        y_d = fy.derivative(1)(n)
        y_dd = fy.derivative(2)(n)
        curvature = (x_d*y_dd - y_d*x_dd) / np.power(x_d** 2 + y_d** 2, 3 / 2)
        return  curvature
    
    
    curvatures_fit = np.array([_get_curvature(skel) for skel in skeletons])
    return curvatures_fit


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
def get_posture_features(skeletons, curvature_window = 4):
    
    
    head_tail_dist = get_head_tail_dist(skeletons)
    quirkiness, major_axis, minor_axis = get_quirkiness(skeletons)
    
    #I prefer to explicity recalculate the lengths, just to do not have to pass the length information
    lengths = get_length(skeletons)
    curvature = get_curvature(skeletons, curvature_window, lengths=lengths)
    eigen_projections = get_eigen_projections(skeletons)
    
    #repack into an ordered dictionary
    data = OrderedDict(
        [
        ('lengths' , lengths),
        ('head_tail_distance' , head_tail_dist),
        ('quirkiness' , quirkiness),
        ('major_axis' , major_axis),
        ('minor_axis' , minor_axis)
        ]
    )
    
    for p in curvature:
        data['curvature_' + p] = curvature[p]
        
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
    