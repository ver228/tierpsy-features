#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:18:13 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables
import tqdm
import warnings
import matplotlib.pylab as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages

EIGENWORMS_COMPONENTS = np.load('./results/ipca_components.npy')

def _h_angles(skeletons):
    '''
    Get skeletons angles
    '''
    dd = np.diff(skeletons, axis=1)
    angles = np.arctan2(dd[..., 0], dd[..., 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1)

    mean_angles = np.mean(angles, axis=1)
    
    angles -= mean_angles[:, None]
    
    return angles, mean_angles


def _h_eigenworms(angles, n_components):
    eigenworms = np.dot(angles, EIGENWORMS_COMPONENTS[:n_components].T)
    return eigenworms


def _h_eigenworms_T(skeletons, n_components = 6):
    '''
    Fully transform the worm skeleton using its eigen components
    '''
    angles, mean_angles = _h_angles(skeletons)
    eigenworms = _h_eigenworms(angles, n_components)
    
    mean_angles = np.unwrap(mean_angles)
    delta_ang = np.hstack((0, np.diff(mean_angles)))
    
    #get how much the head position changes over time but first rotate it to the skeletons to 
    #keep the same frame of coordinates as the mean_angles first position
    ang_m = mean_angles[0]
    R = np.array([[np.cos(ang_m), -np.sin(ang_m)], [np.sin(ang_m), np.cos(ang_m)]])
    head_r = skeletons[:, 0, :]
    head_r = np.dot(R, (head_r - head_r[0]).T)
    delta_xy = np.vstack((np.zeros((1,2)), np.diff(head_r.T, axis=0)))
    
    #size of each segment (the mean is a bit optional, at this point all the segment should be of equal size)
    segment_l = np.mean(np.linalg.norm(np.diff(skeletons, axis=1), axis=2), axis=1)
    
    #pack all the elments of the transform
    DT = np.hstack((delta_xy, segment_l[:, None], delta_ang[:, None], eigenworms))
    
    return DT


def _h_center_skeleton(skeletons, body_range = (8, 41)):
    body_coords = np.mean(skeletons[:, body_range[0]:body_range[1] + 1, :], axis=1)
    return skeletons - body_coords[:, None, :]



with open('./results/PCA_errors_SWDB.pkl', 'rb') as fid:
    all_errors = pickle.load(fid)

dd = [(k, v[2].max()) for k,v in all_errors.items()]
strains, _ =  zip(*sorted(dd, key = lambda x: x[1], reverse=True))


set_type = 'SWDB'
main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/{0}/{0}_skel_smoothed.hdf5'.format(set_type)

with pd.HDFStore(main_dir, 'r') as fid:
    skels_ranges = fid['/skeletons_groups']

skel_g = skels_ranges.groupby('strain')
#for i_strain, (strain, dat) in enumerate(skel_g):

    
if False:
    for strain in strains[:10]:
        pdf = PdfPages('./results/skeletons_errors/{}_skel_errors.pdf'.format(strain))
        
        dat = skel_g.get_group(strain)
        with tables.File(main_dir, 'r') as fid:
            skel_node = fid.get_node('/skeletons_data')
            is_bad_node = fid.get_node('/is_bad_skeleton')
            
            tot = 0
            dd = tqdm.tqdm(enumerate(dat.iterrows()), total=len(dat), desc = strain)
            for ichuck, (irow, row) in dd:
                
                ini = row['ini']
                fin = row['fin']
                #is_bad_skeleton = is_bad_node[ini:fin+1]>0
                
                skels = skel_node[ini:fin+1, :, :]
                #skels = skels[~is_bad_skeleton]
                fig = plt.figure(figsize=(15, 3))
                for nn in range(4,9):
                    aa, am = _h_angles(skels)
                    pca_r = EIGENWORMS_COMPONENTS[:nn]
                    DD = np.dot(aa, pca_r.T)
                    DD_a = np.dot(DD, pca_r)    
                    
                    err = np.abs(DD_a-aa).mean(axis=0)
                    
                    ind = np.argmax(err)
                    segment_l = np.mean(np.linalg.norm(np.diff(skels, axis=1), axis=2), axis=1)
                    
                    aa_r = DD_a + am[:, None]
                    
                    xx = np.cos(aa_r)*segment_l[:, None]
                    xx = np.hstack((skels[..., 0, 1][:, None],  xx))
                    xx = np.cumsum(xx, axis=1) 
                    
                    yy = np.sin(aa_r)*segment_l[:, None]
                    yy = np.hstack((skels[..., 0, 0][:, None],  yy))
                    yy = np.cumsum(yy, axis=1)
                    
                    
                    ax = plt.subplot(1,5,nn-3)
                    plt.plot(xx[ind], yy[ind], 'o')
                    plt.plot(skels[ind, : , 1], skels[ind, : , 0], '.')
                    plt.title(nn)
                    plt.axis('equal')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    
                pdf.savefig(fig)
                plt.close()
            pdf.close()


if True:
    strain = 'N2'
    dat = skel_g.get_group(strain)
    tot_rows = (dat['fin']-dat['ini']+1).sum()
    
    #eigenworm_transforms = np.full((tot_rows, 10), np.nan)
    collected_data = np.full((tot_rows, 98), np.nan)
    with tables.File(main_dir, 'r') as fid:
        skel_node = fid.get_node('/skeletons_data')
        is_bad_node = fid.get_node('/is_bad_skeleton')
        
        
        tot = 0
        dd = tqdm.tqdm(enumerate(dat.iterrows()), total=len(dat), desc = strain)
        for ichuck, (irow, row) in dd:
            
            ini = row['ini']
            fin = row['fin']
            skels = skel_node[ini:fin+1, :, :]
            #DT = _h_eigenworms_T(skels)
            
            #DT, _ = _h_angles(skels)
            DT = _h_center_skeleton(skels)
            DT = DT.reshape((DT.shape[0], -1))
            
            n_samples = DT.shape[0]
            collected_data[tot:tot + n_samples] = DT
            #eigenworm_transforms[tot:tot + n_samples] = DT
            tot += n_samples 
            
            
    
    Q = np.percentile(collected_data, [2, 50, 98], axis=0)
    
    for q in Q.T:
        print(q[1], q[2]-q[0])
    #%%
if False:
    delta_y = DT[:, 0]
    delta_x = DT[:, 1]
    seg_l = DT[:, 2]
    delta_ang = DT[:, 3]
    
    eigenworms = DT[:, 4:]
    
    xx = np.cumsum(delta_x)
    yy = np.cumsum(delta_y)
    
    mean_angles = np.cumsum(delta_ang)
    
    n_components = eigenworms.shape[1]
    angles = np.dot(eigenworms, EIGENWORMS_COMPONENTS[:n_components])
    angles += mean_angles[:, None]
    
    #angles += am[0]
    
    ske_x = np.cos(angles)*seg_l[:, None]
    ske_x = np.hstack((xx[:, None],  ske_x))
    #ske_x = np.hstack((np.zeros((xx.size,1)),  ske_x))
    ske_x = np.cumsum(ske_x, axis=1) 
    
    ske_y = np.sin(angles)*seg_l[:, None]
    ske_y = np.hstack((yy[:, None],  ske_y))
    #ske_y = np.hstack((np.zeros((yy.size,1)),  ske_y))
    
    ske_y = np.cumsum(ske_y, axis=1) 
    
    skels_n = np.concatenate((ske_y[..., None], ske_x[..., None]), axis=1)
    
    n_i = 1000
    plt.figure()
    plt.plot(ske_x[:n_i].T, ske_y[:n_i].T, 'g')
    #skels_m = skels - skels[:, 0, :][:, None, :]
    skels_m = skels - skels[0, 0, :]
    plt.plot(skels_m[:n_i, :, 1].T, skels_m[:n_i,:, 0].T, 'r')
    plt.axis('equal')
    #plt.plot(skels[100:200:10, :, 0].T, skels[100:200:10,:, 1].T)
    #%%
    