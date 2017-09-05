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
    ts = np.arange(1000)
    x = np.sin(ts*2*np.pi/5) + np.sin(ts*2*np.pi/25)
    valid, = np.where(~np.isnan(x))
    
    t_i = np.hstack([-1, valid, x.shape[0]])
    x_i = np.pad(x[valid], (1,1), 'edge')
    f = interp1d(t_i, x_i)
    
    ts = np.arange(x.shape[0])
    xs = f(ts)
     
    
    #%%
    import pycwt as wavelet
    
    dt = 1
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 10 / dj  # Seven powers of two with dj sub-octaves
    #alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    mother = wavelet.Morlet(6)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(xs, dt, dj, s0, J,
                                                      mother)
    plt.imshow(np.abs(wave)**2, aspect='auto')
    