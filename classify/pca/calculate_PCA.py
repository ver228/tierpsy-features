#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:51:32 2017

@author: ajaver
"""
import warnings
import tables 
import pandas as pd
import numpy as np
import tqdm
import os
from sklearn.decomposition import PCA, IncrementalPCA


#main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/CeNDR/CeNDR_skel_smoothed.hdf5'
main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/SWDB/SWDB_skel_smoothed.hdf5'

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

def get_ipca(main_dir, batch_n = 500000):
    ipca = IncrementalPCA()
    
    with tables.File(main_dir, 'r') as fid:
        is_bad_skeleton = fid.get_node('/is_bad_skeleton')[:]
        valid_ind = np.where(is_bad_skeleton==0)[0]
        del is_bad_skeleton
        
             
        skel_node = fid.get_node('/skeletons_data')
        for ii in tqdm.trange(0, valid_ind.size, batch_n): 
            ind = valid_ind[ii:ii+batch_n]
            skels = skel_node[ind, :, :]
            angs = _h_angles(skels)
            ipca.partial_fit(angs)
    
    return ipca
    

def get_pca_per_strain(main_dir):
    with pd.HDFStore(main_dir, 'r') as fid:
        skels_ranges = fid['/skeletons_groups']
    
    skel_g = skels_ranges.groupby('strain')
    
    
    all_pca = {}
    for ii, (strain, dat) in enumerate(skel_g):
        print(ii, strain)
        tot_rows = (dat['fin']-dat['ini']+1).sum()
        angs = np.full((tot_rows, 48), np.nan, dtype = np.float32)
        
        with tables.File(main_dir, 'r') as fid:
            skel_node = fid.get_node('/skeletons_data')
            is_bad_node = fid.get_node('/is_bad_skeleton')
            
            tot = 0
            for irow, row in tqdm.tqdm(dat.iterrows(), total=len(dat), desc = strain):
                
                ini = row['ini']
                fin = row['fin']
                is_bad_skeleton = is_bad_node[ini:fin+1]
                
                valid_ind = np.where(is_bad_skeleton==0)[0] + ini
                aa = _h_angles(skel_node[valid_ind, :, :])
                
                angs[tot:tot+aa.shape[0]] = aa
                tot += aa.shape[0]
        
        pca = PCA()
        pca.fit(angs)
        all_pca[strain] = pca.components_

    return all_pca

from sklearn.decomposition import FastICA
import multiprocessing as mp
from threading import Thread


class ProducerThread(Thread):
    def run(self):
        global queue
        
        with pd.HDFStore(main_dir, 'r') as fid:
            skels_ranges = fid['/skeletons_groups']
    
        skel_g = skels_ranges.groupby('strain')
        
        
        for ii, (strain, dat) in enumerate(skel_g):
            print(ii, strain)
            tot_rows = (dat['fin']-dat['ini']+1).sum()
            angs = np.full((tot_rows, 48), np.nan, dtype = np.float32)
            
            with tables.File(main_dir, 'r') as fid:
                skel_node = fid.get_node('/skeletons_data')
                is_bad_node = fid.get_node('/is_bad_skeleton')
                tot = 0
                for irow, row in tqdm.tqdm(dat.iterrows(), total=len(dat), desc = strain):
                    
                    ini = row['ini']
                    fin = row['fin']
                    is_bad_skeleton = is_bad_node[ini:fin+1]
                    
                    valid_ind = np.where(is_bad_skeleton==0)[0] + ini
                    aa = _h_angles(skel_node[valid_ind, :, :])
                    
                    angs[tot:tot+aa.shape[0]] = aa
                    tot += aa.shape[0]
                
            
            queue.put((strain, angs))


def _h_get_ica(dat):
    strain, X = dat
    ica = FastICA(max_iter=1000)
    ica.fit(X)
    return (strain, ica)
    
if __name__ == '__main__':
    
    #ipca = get_ipca(main_dir)
    #all_pca = get_pca_per_strain(main_dir)
    n_process = 10
    queue = mp.Queue(n_process)
    ProducerThread().start()
    
    p = mp.Pool(n_process)
    dd = p.map(_h_get_ica, iter(queue.get, None))
    ica_results = list(dd)
    #%%
    
        
        