#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:40:48 2017

@author: ajaver
"""
import numpy as np
from tierpsy_features.postures import get_posture_features
from tierpsy_features.velocities import get_velocity_features, _h_get_velocity
from scipy.interpolate import interp1d

import pywt
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    data = np.load('../notebooks/data/worm_example_small_W1.npz')
    #data = np.load('../notebooks/data/worm_example.npz')
    
    skeletons = data['skeleton']
    dorsal_contours = data['dorsal_contour']
    ventral_contours = data['ventral_contour']
    widths = data['widths']
    
    fps = 25
    delta_time = 1/3 #delta time in seconds to calculate the velocity
    delta_frames = int(round(fps*delta_time))
    
    feat_posture = get_posture_features(skeletons, curvature_window = 4)
    velocities = get_velocity_features(skeletons, delta_time, fps)
    
    
    #%%
    #x = feat_posture['curvature_midbody'].values
    #x = velocities['speed'].values
    x = np.sin(ts/5)# + np.sin(ts/5)
    valid, = np.where(~np.isnan(x))
    
    t_i = np.hstack([-1, valid, x.shape[0]])
    x_i = np.pad(x[valid], (1,1), 'edge')
    f = interp1d(t_i, x_i)
    
    ts = np.arange(x.shape[0])
    xs = f(ts)
    
    widths = np.arange(1, 125)
    cwtmatr, freqs = pywt.cwt(xs, widths, 'morl')
    
    #cA, cD = pywt.dwt(xs, 'haar')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(np.abs(cwtmatr[:, :2000])**2, interpolation='none')
    plt.subplot(2,1,2)
    plt.plot(x[:2000])
    
    plt.figure()
    plt.plot(np.max(cwtmatr,0))
    
    #%%
#    window_size = 500
#    wh = np.hamming(window_size)
#    
#    dat = []
#    for ii in range(xs.size-window_size):
#        ws = xs[ii: ii + window_size]*wh
#        dd = np.fft.rfft(ws)
#        dat.append(np.abs(dd))
#    dat = np.array(dat).T
    #plt.subplot(3,1,3)
    #plt.imshow(np.log10(dat**2)[:, :2000], interpolation='none')
    
    #plt.plot(np.abs(np.fft.rfft(ws)))
    
    #np.fft.rfft(x)
    #%%
    x = feat_posture['curvature_midbody'].values
    delta_frames= int(round(fps*delta_time))
    x_v = _h_get_velocity(x, delta_frames, fps)
    tt = np.arange(x_v.size)
    plt.subplot(2,1,1)
    plt.plot(tt, x_v)
        
    plt.subplot(2,1,1)
    plt.xlim(3000,3500)
    
    plt.subplot(2,1,2)
    plt.plot(x)
    plt.xlim(3000,3500)