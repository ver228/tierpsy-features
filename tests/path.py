#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:54:00 2017

@author: ajaver
"""
import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from tierpsy_features.helper import DataPartition

def _curvature_fun(x_d, y_d, x_dd, y_dd):
    return (x_d*y_dd - y_d*x_dd)/(x_d*x_d + y_d*y_d)**1.5

def _h_curvature_grad(skeletons, points_window=1, length=None):
    
    if points_window is None:
        points_window = 1
    
    if skeletons.shape[0] <= points_window*2:
        return np.full((skeletons.shape[0], skeletons.shape[1]), np.nan)
    
    #this is less noisy than numpy grad
    def _gradient_windowed(X, points_window):
        w_s = 2*points_window
        right_side = np.pad(X[:-w_s, :], ((w_s, 0), (0,0)), 'edge')
        left_side = np.pad(X[w_s:, :], ((0, w_s), (0,0)), 'edge')
    
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
    
    gx = d[..., 0]
    gy = d[..., 1]
    
    ggx = dd[..., 0]
    ggy = dd[..., 1]
    
    return _curvature_fun(gx, gy, ggx, ggy)

def _h_path_curvature(skeletons, path_step = 25, points_window = 10):
    #%%
    p_obj = DataPartition(n_segments=skeletons.shape[1])
    
    body_coords = p_obj.apply(skeletons, 'body', func=np.mean)
    
    
    xx = body_coords[:,0]
    yy = body_coords[:,1]
    tt = np.arange(body_coords.shape[0])
    
    good = ~np.isnan(xx)
    
    x_i = xx[good] 
    y_i = yy[good] 
    t_i = tt[good]
    
    t_i = np.hstack([-1, t_i, body_coords.shape[0]]) 
    x_i = np.hstack([x_i[0], x_i, x_i[-1]]) 
    y_i = np.hstack([y_i[0], y_i, y_i[-1]]) 
    
    fx = interp1d(t_i, x_i)
    fy = interp1d(t_i, y_i)
    
    xx_i = fx(tt)
    yy_i = fy(tt)
    
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(xx_i)
    dy = np.diff(yy_i)
    dr = np.sqrt(dx * dx + dy * dy)
    
    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))
    #%%
    fx = interp1d(lengths, xx_i)
    fy = interp1d(lengths, yy_i)
    ft = interp1d(lengths, tt)
    
    sub_lengths = np.arange(lengths[0], lengths[-1], path_step)
    xs = fx(sub_lengths)
    ys = fy(sub_lengths)
    ts = ft(sub_lengths)
     
    curve = np.vstack((xs, ys)).T
    #%%
    #this is less noisy than numpy grad
    def _gradient_windowed(X, points_window):
        
        w_s = 2*points_window
        right_side = np.pad(X[:-w_s, :], ((w_s, 0), (0,0)), 'edge')
        left_side = np.pad(X[w_s:, :], ((0, w_s), (0,0)), 'edge')
    
        ramp = np.full(X.shape[0] - w_s, w_s*2)
        ramp = np.pad(ramp, 
                        pad_width = (points_window, points_window), 
                        mode='linear_ramp',  
                        end_values = w_s
                        )
        grad = (left_side - right_side) / ramp[:, None]
        
        return grad
    d = _gradient_windowed(curve, 1)
    dd = _gradient_windowed(d, 1)
    gx = d[..., 0]
    gy = d[..., 1]
    
    ggx = dd[..., 0]
    ggy = dd[..., 1]
    
    curvature_r =  _curvature_fun(gx, gy, ggx, ggy)
    
    #%%
#    s_center = curve[points_window:-points_window] #center points
#    s_left = curve[:-2*points_window] #left side points
#    s_right = curve[2*points_window:] #right side points
#    
#    
#    d_left = s_left - s_center 
#    d_right = s_center - s_right
#    
#    #arctan2 expects the y,x angle
#    ang_l = np.arctan2(d_left[...,1], d_left[...,0])
#    ang_r = np.arctan2(d_right[...,1], d_right[...,0])
#    ang = np.unwrap(ang_r)-np.unwrap(ang_l)
#    
#    curvature_r = ang/(path_step*2*points_window)
    
    #curvature_r = _h_curvature_grad(curve)
    
    ts_i = np.hstack((-1, ts, tt[-1] + 1))
    c_i = np.hstack((curvature_r[0], curvature_r, curvature_r[-1]))
    curvature_t = interp1d(ts_i, c_i)(tt)
    
    return curvature_t, body_coords
#%%
if __name__ == '__main__':
    import os
    
    base_dir = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/singleworm_skels'

    #data = np.load(os.path.join(base_dir, 'worm_example_small_W1.npz'))
    data = np.load(os.path.join(base_dir, 'worm_example.npz'))
    skeletons = data['skeleton']
    ventral_contour = data['ventral_contour']
    dorsal_contour = data['dorsal_contour']
    
    path_curvature, body_coords = \
        _h_path_curvature(skeletons, path_step = 25, points_window = 10)


    
    #%%
    import matplotlib.pylab as plt
    from matplotlib.collections import LineCollection
    
    path_curvature = np.clip(path_curvature, -0.01, 0.01)
    
    curv_range = (np.nanmin(path_curvature), np.nanmax(path_curvature))
    
    points = body_coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, 
                        cmap = plt.get_cmap('plasma'),
                        norm = plt.Normalize(*curv_range))
    lc.set_array(path_curvature)
    lc.set_linewidth(2)
    
    fig2 = plt.figure()
    plt.gca().add_collection(lc)
    
    plt.xlim(3000, 11000)
    plt.ylim(3000, 11000)
    #%%
    pix2microns = 50
    
    x_min = np.nanmin(ventral_contour[:, :, 0])
    x_max = np.nanmax(ventral_contour[:, :, 0])
    
    y_min = np.nanmin(ventral_contour[:, :, 1])
    y_max = np.nanmax(ventral_contour[:, :, 1])
    
    
    rx = int(round((x_max - x_min)/pix2microns))
    ry = int(round((y_max - y_min)/pix2microns))
    
    size_counts = (rx + 1, ry + 1)
    
    
    #%%
    partitions_dflt = {'head': (0, 8),
                            'neck': (8, 16),
                            'midbody': (16, 33),
                            'hips': (33, 41),
                            'tail': (41, 49),
                            'all': (0, 49),
                            'body': (8, 41)
                            }
    
    all_cnts = {}
    for part, rr in partitions_dflt.items():
        
        p_vc = ventral_contour[:, rr[0]:rr[1], :].astype(np.float32)
        p_dc = dorsal_contour[:, rr[0]:rr[1], :].astype(np.float32)
        h = np.hstack((p_vc[:, ], p_dc[:, ::-1, :], p_vc[:, 0, :][:, None, :]))
        
        
        cnts = [np.round((x-np.array((x_min, y_min))[None, :])/pix2microns) for x in h]
        
        
        counts = np.zeros(size_counts, np.float32)
        #%%
        for ii, cnt in enumerate(cnts):
            if np.any(np.isnan(cnt)):
                continue
            cc = np.zeros(size_counts, np.float32)
            cc = cv2.drawContours(cc, [cnt[:, None, :].astype(np.int)], -1, 1)
            counts += cc
            
        
        #%%
        
        plt.figure()
        plt.imshow(counts)
        plt.title(part)
        
        all_cnts[part] = counts
    
        print(part)
    #%%
    plt.figure()
    tt = 2500
    plt.plot(ventral_contour[tt,:,0], ventral_contour[tt,:,1])
    plt.plot(dorsal_contour[tt,:,0], dorsal_contour[tt,:,1])
    plt.axis('equal')