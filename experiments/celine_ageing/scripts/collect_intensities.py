#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:11:40 2017

@author: ajaver
"""
import os
import tables
import pandas as pd
import numpy as np
import cv2

if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    save_dir = './int_avg_maps'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir_s = './int_summary'
    if not os.path.exists(save_dir_s):
        os.makedirs(save_dir_s)
    
    exp_file = 'ageing_celine.csv'
    experiments_df = pd.read_csv(exp_file)
    
    bad_f = []
    tot_exp = 0
    for d_id, dat in experiments_df.groupby(('replicated_n', 'strain', 'worm_id')):
        print(d_id)
        
        s_intensities = []
        for irow, row in dat.iterrows():
            fname = os.path.join(row['directory'], row['base_name'] + '_intensities.hdf5')
            
            if not os.path.exists(fname):
                continue
            
            try:
                with tables.File(fname, 'r') as fid:
                    if not '/straighten_worm_intensity' in fid:
                        continue
                    straighten_worm_intensity = fid.get_node('/straighten_worm_intensity')[:]
                    if straighten_worm_intensity.shape[0] < 1000:
                        continue
                    
            except:
                bad_f.append(fname)
                    
            s_int = np.median(straighten_worm_intensity, axis=0).astype(np.uint8)
            if row['ventral_orientation'] == 'CCW':
                s_int = s_int[:, ::-1]
            
            save_name = os.path.join(save_dir, row['base_name'] +  '_map.png')
            cv2.imwrite(save_name, s_int)
            
            s_intensities.append((row['day'], s_int))
        
        
        tot = len(s_intensities)
        plt.figure(figsize=(1*tot, 7))
        for ii, (f, s_int) in enumerate(s_intensities):
            ax=plt.subplot(1, tot, ii + 1)
            plt.imshow(s_int)
            plt.title(f)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
        strT = 'S{}_R{}_W{}'.format(row['strain'], row['replicated_n'], row['worm_id'])
        plt.suptitle(strT)
        plt.savefig(os.path.join(save_dir_s, strT +  '.png'))