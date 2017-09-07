#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:59:23 2017

@author: ajaver
"""

import numpy as np
import warnings
import pandas as pd

from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from tierpsy_features.helper import nanunwrap
from tierpsy_features.postures import get_length

def _h_curvature_angles(skeletons, window_length = None, lengths=None):
    if window_length is None:
        window_length = 7

    points_window = int(round(window_length/2))
    
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

def _curvature_fun(x_d, y_d, x_dd, y_dd):
    return (x_d*y_dd - y_d*x_dd)/(x_d*x_d + y_d*y_d)**1.5

def _h_curvature_savgol(skeletons, window_length = None, length=None):
    '''
    Calculate the curvature using univariate splines. This method is slower and can fail
    badly if the fit does not work, so I am only using it as testing
    '''

    if window_length is None:
        window_length = 7

    def _fitted_curvature(skel):
        if np.any(np.isnan(skel)):
            return np.full(skel.shape[0], np.nan)
        
        x = skel[:, 0]
        y = skel[:, 1]

        x_d = savgol_filter(x, window_length=window_length, polyorder=3, deriv=1)
        y_d = savgol_filter(y, window_length=window_length, polyorder=3, deriv=1)
        x_dd = savgol_filter(x, window_length=window_length, polyorder=3, deriv=2)
        y_dd = savgol_filter(y, window_length=window_length, polyorder=3, deriv=2)
        curvature = _curvature_fun(x_d, y_d, x_dd, y_dd)
        return  curvature

    
    curvatures_fit = np.array([_fitted_curvature(skel) for skel in skeletons])
    return curvatures_fit


def _h_curvature_spline(skeletons, points_window=None, length=None):
    '''
    Calculate the curvature using univariate splines. This method is slower and can fail
    badly if the fit does not work, so I am only using it as testing
    '''

    def _spline_curvature(skel):
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

        curvature = _curvature_fun(x_d, y_d, x_dd, y_dd)
        return  curvature

    
    curvatures_fit = np.array([_spline_curvature(skel) for skel in skeletons])
    return curvatures_fit

def _h_curvature_grad(skeletons, points_window=1, length=None):
    
    if points_window is None:
        points_window = 1
    
    if skeletons.shape[0] <= points_window*2:
        return np.full((skeletons.shape[0], skeletons.shape[1]), np.nan)
    
    #this is less noisy than numpy grad
    def _gradient_windowed(X, points_window):
        w_s = 2*points_window
        right_side = np.pad(X[:, :-w_s, :], ((0,0), (w_s, 0), (0,0)), 'edge')
        left_side = np.pad(X[:, w_s:, :], ((0,0), (0, w_s), (0,0)), 'edge')
    
        ramp = np.full(X.shape[1] - w_s, w_s*2)
        ramp = np.pad(ramp, 
                        pad_width = (points_window, points_window), 
                        mode='linear_ramp',  
                        end_values = w_s
                        )
        grad = (left_side - right_side) / ramp[None, :, None]
        
        return grad
    
    d = _gradient_windowed(skeletons, points_window)
    dd = _gradient_windowed(d, points_window)
    
    gx = d[:, :, 0]
    gy = d[:, :, 1]
    
    ggx = dd[:, :, 0]
    ggy = dd[:, :, 1]
    
    return _curvature_fun(gx, gy, ggx, ggy)
    

def get_curvature_features(skeletons, method = 'grad', points_window=None, lengths=None):
    curvature_funcs = {
            'angle' : _h_curvature_angles, 
            'spline' : _h_curvature_spline, 
            'savgol' : _h_curvature_savgol,
            'grad' : _h_curvature_grad
            }
    
    
    assert method in curvature_funcs
    
    
    
    if method == 'angle':
        segments_ind_dflt = {
            'head' : 0,
            'neck' : 0.25,
            'midbody' : 0.5, 
            'hips' : 0.75,
            'tail' : 1.,
        }
    else:
        segments_ind_dflt = {
            'head' : 5/48,
            'neck' : 15/48,
            'midbody' : 24/48, 
            'hips' : 33/48,
            'tail' : 45/48,
        }
    
    curvatures = curvature_funcs[method](skeletons, points_window, lengths)
    max_angle_index = curvatures.shape[-1]-1
    segments_ind = {k:int(round(x*max_angle_index)) for k,x in segments_ind_dflt.items()}
    
    curv_dict = {'curvature_' + x :curvatures[:, ind] for x,ind in segments_ind.items()}
    data = pd.DataFrame.from_dict(curv_dict)
    
    return data
    
    