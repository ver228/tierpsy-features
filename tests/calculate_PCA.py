#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:51:32 2017

@author: ajaver
"""
import warnings
import tables 
import numpy as np
import multiprocessing as mp
import tqdm

main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/CeNDR/CeNDR_skel_smoothed.hdf5'
#main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/SWDB/SWDB_skel_smoothed_v2.hdf5'

def _h_angles(skeletons):
    '''
    Get skeletons angles
    '''
    dd = np.diff(skeletons,axis=1);
    angles = np.arctan2(dd[...,0], dd[...,1])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1);
    
    mean_angles = np.mean(angles, axis=1)
    angles -= mean_angles[:, None]
    
    return angles

def gen_skels(main_dir, batch_n = 5000):
    with tables.File(main_dir, 'r') as fid:
        is_bad_skeleton = fid.get_node('/is_bad_skeleton')[:]
        valid_ind = np.where(is_bad_skeleton==0)[0]
        del is_bad_skeleton
        
             
        skel_node = fid.get_node('/skeletons_data')
        for ii in tqdm.trange(0, valid_ind.size, batch_n): 
            ind = valid_ind[ii:ii+batch_n]
            skeletons_data = skel_node[ind, :, :]
            yield(skeletons_data)
    

    
if __name__ == '__main__':
    p = mp.Pool(5)
    gen = gen_skels(main_dir, batch_n = 50000)
    all_ang = p.map(_h_angles, gen)
        
        