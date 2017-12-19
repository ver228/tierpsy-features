#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:07:53 2017

@author: ajaver
"""

import os
from skimage.filters import rank
from scipy.stats import entropy
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import cv2
import tables

if __name__ == '__main__':
    s_int_dir = './int_avg_maps'
    if not os.path.exists(s_int_dir):
        os.makedirs(s_int_dir)
    
    exp_file = 'ageing_celine.csv'
    experiments_df = pd.read_csv(exp_file)
    
    tot_exp = 0
    for d_id, dat in experiments_df.groupby(('replicated_n', 'strain', 'worm_id')):
        print(d_id)
        s_intensities = []
        for irow, row in dat.iterrows():
            fname = os.path.join(s_int_dir, row['base_name'] +  '_map.png')
            
            if not os.path.exists(fname):
                continue
                    
            s_int = cv2.imread(fname, cv2.IMREAD_UNCHANGED )
            
            s_intensities.append((row['day'], s_int))
        
        if len(s_intensities) == 0:
            continue
        #%%
        tot = len(s_intensities)
        plt.figure(figsize=(1*tot, 7))
        for ii, (f, s_int) in enumerate(s_intensities):
            ax=plt.subplot(1, tot, ii + 1)
            plt.imshow(s_int, interpolation='none')
            plt.title(f)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
        strT = 'S{}_R{}_W{}'.format(row['strain'], row['replicated_n'], row['worm_id'])
        plt.suptitle(strT)
        
        #%%
        ss = []
        for day, s_int in s_intensities:
            pk = np.bincount(s_int[10:-10, 4:-4].flatten(), minlength=255)
            sk = entropy(pk)
            
            ss.append((day, sk))
        
        days, sk = zip(*ss)            
        #plt.figure(figsize=(10,5))
        #plt.plot(days, sk, 'o')
        
        from skimage.feature import greycomatrix
        
        ss = []
        for day, s_int in s_intensities:
            glcm = greycomatrix(s_int[:, 5:-5], [3], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
            p = glcm.flatten() + 1.e-8
            sk_d = np.sum(p*np.log(p))
            ss.append((day, sk_d))
        
        days, sk_d = zip(*ss)            
        
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(days, sk, 'o')
        plt.subplot(2,1,2)
        plt.plot(days, sk_d, 'o')
        
        #%%
#        #%%
#        dat = []
#        tot = len(s_intensities)
#        plt.figure(figsize=(1*tot, 7))
#        for ii, (f, s_int) in enumerate(s_intensities):
#            d_map = s_int.astype(np.float32) - s_intensities[1][1]
#            ax=plt.subplot(1, tot, ii + 1)
#            plt.imshow(d_map, interpolation='none')
#            plt.title(f)
#            ax.xaxis.set_visible(False)
#            ax.yaxis.set_visible(False)
#            dat.append(np.abs(d_map).mean())
#            
#        strT = 'S{}_R{}_W{}'.format(row['strain'], row['replicated_n'], row['worm_id'])
#        plt.suptitle(strT)
#        
#        
#        
#        plt.figure()
#        plt.plot(dat)
#        #%%
##        tot = len(s_intensities)
##        plt.figure(figsize=(7, 1*tot))
##        for ii, (day, s_int) in enumerate(s_intensities):
##            ax=plt.subplot(tot, 1, ii + 1)
##            dd = rank.entropy(s_int, np.ones((5,21)))
##            plt.plot(dd[7:-7, 7])
##            #plt.imshow(rank.entropy(s_int, np.ones((15,15))))
##        
##        #%%
##        plt.figure()
##        ss = []
##        days = []
##        for ii, (day, s_int) in enumerate(s_intensities):
##            dd = rank.entropy(s_int, np.ones((12,5)))
##            pp = np.percentile(dd[10:-10, 7], q = [50])
##            ss.append(pp)
##            days.append(day)
##        ss = np.array(ss)
##        plt.figure()
##        plt.plot(days, ss, 'o')
        #%%
        tot_exp += 1
        print(tot_exp)
        if tot_exp > 10:
            
            break