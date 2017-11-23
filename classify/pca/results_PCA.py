#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:23:52 2017

@author: ajaver
"""
import tables
import pickle
import numpy as np

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import tqdm
from calculate_PCA import _h_angles#, main_dir

from tierpsy_features import EIGEN_PROJECTION_FILE


with tables.File(EIGEN_PROJECTION_FILE) as fid:
    old_pca = fid.get_node('/eigenWorms')[:]
    old_pca = -old_pca.T

all_pca = np.load('ipca_components.npy')

with open('strain_pca.pkl', 'rb') as fid:
    strain_pca = pickle.load(fid)

#%%
if False:
    with PdfPages('PCA_results.pdf') as pdf:
        for kk in range(48):
            print(kk)
            fig = plt.figure()
            
            main_v = all_pca[kk]
            
            for k,v in strain_pca.items():
                if np.sum(np.abs(main_v + v[kk])) < np.sum(np.abs(main_v - v[kk])):
                    v[kk] *= -1
                
                if k != 'N2':
                    plt.plot(v[kk], color='lightgray')
                else:
                    plt.plot(v[kk], color='lightgray', label = 'by strain')
            
            plt.plot(main_v, 'r', label = 'Incremental PCA', lw=2)
            
            if kk < old_pca.shape[0]:
                plt.plot(old_pca[kk], 'k', label = 'Old', lw=2)
            plt.legend()
            
            plt.title('PCA {}'.format(kk + 1))
            
            pdf.savefig(fig)
            plt.close()
        


#set_type = 'CeNDR'
set_type = 'SWDB'

main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/{0}/{0}_skel_smoothed.hdf5'.format(set_type)

pdf = PdfPages('PCA_errors_{}.pdf'.format(set_type))
with pd.HDFStore(main_dir, 'r') as fid:
    skels_ranges = fid['/skeletons_groups']

skel_g = skels_ranges.groupby('strain')
    
    
all_errors = {}


for i_strain, (strain, dat) in enumerate(skel_g):
    
    print(i_strain, strain)
    tot_rows = (dat['fin']-dat['ini']+1).sum()
    angs = np.full((tot_rows, 48), np.nan, dtype = np.float32)
    
    with tables.File(main_dir, 'r') as fid:
        skel_node = fid.get_node('/skeletons_data')
        is_bad_node = fid.get_node('/is_bad_skeleton')
        
        tot = 0
        for irow, row in tqdm.tqdm(dat.iterrows(), total=len(dat), desc = strain):
            
            ini = row['ini']
            fin = row['fin']
            is_bad_skeleton = is_bad_node[ini:fin+1] > 0
            skels = skel_node[ini:fin+1, :, :]
            aa = _h_angles(skels[~is_bad_skeleton])
            
            angs[tot:tot+aa.shape[0]] = aa
            tot += aa.shape[0]
    
    

    pca_errors = []
    
    n_components = np.arange(4, 12)
    
    for nn in n_components:
        print(nn)
        pca_r = all_pca[:nn]
        DD = np.dot(angs, pca_r.T)
        DD_a = np.dot(DD, pca_r)    
        err = np.abs(DD_a-angs).mean(axis=0)
        pca_errors.append(err)
    
    pca_errors = np.vstack(pca_errors)
    
    #%%
    fig = plt.figure(figsize=(12, 6))
    for ii, nn in enumerate(n_components):
        plt.subplot(2,4, ii+1)
        plt.plot(pca_errors[ii])
        plt.ylim([0, 0.3])
        plt.title('{} PC'.format(nn))
        plt.xlim((-1, 49))
    plt.suptitle(strain)
    
    pdf.savefig(fig)
    plt.close()
    
    
    all_errors[strain] = pca_errors
    
    
pdf.close()
#%%
with open('PCA_errors_{}.pkl'.format(set_type), 'wb') as fid:
    pickle.dump(all_errors, fid)
#%%
with PdfPages('PCA_errors_{}_all.pdf'.format(set_type), 'a') as pdf:

    fig = plt.figure(figsize=(12, 6))
    for ii, nn in enumerate(n_components):
        plt.subplot(2,4, ii+1)
        plt.plot(np.vstack([x[ii] for x in all_errors.values()]).T)
        plt.ylim([0, 0.3])
        plt.title('{} PC'.format(nn))
        plt.xlim((-1, 49))
    plt.suptitle('ALL')
    pdf.savefig(fig)
