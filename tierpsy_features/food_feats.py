#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:20:00 2017

@author: ajaver
"""

import numpy as np
from tierpsy.analysis.feat_create.obtainFeatures import getGoodTrajIndexes
from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormFromTableSimple
import matplotlib.path as mplPath

from find_food import get_food_contour
from tierpsy.helper.params import read_microns_per_pixel



def get_worm_partitions(n_segments=49):
    worm_partitions = { 'head': (0, 8),
                        'neck': (8, 16),
                        'midbody': (16, 33),
                        'hips': (33, 41),
                        'tail': (41, 49),
                        'head_tip': (0, 3),
                        'head_base': (5, 8),
                        'tail_base': (41, 44),
                        'tail_tip': (46, 49),
                        'all': (0, 49),
                        'body': (8, 41)
                        }
    
    if n_segments != 49:
        r_fun = lambda x : int(round(x/49*n_segments))
        for key in worm_partitions:
            worm_partitions[key] = tuple(map(r_fun, worm_partitions[key]))
        
    return worm_partitions

def get_partition_transform(data, func=np.mean, partitions=None):
    n_segments = data.shape[1]
    worm_partitions = get_worm_partitions(n_segments)
    if partitions is None:
        partitions = worm_partitions.keys()
    
    data_transformed = {}
    for pp in partitions:
        ini, fin = worm_partitions[pp]
        data_transformed[pp] = func(data[:, ini:fin, :], axis=1)
    return data_transformed
    
def obtain_food_cnt(mask_video):
    #%%
    microns_per_pixel = read_microns_per_pixel(mask_video)
    circy, circx = get_food_contour(mask_video, is_debug=False)
        #%%
    food_cnt = np.vstack((circx, circy)).T*microns_per_pixel
    #polar coordinates from the centroid
    food_centroid = np.mean(food_cnt, axis=0)
    food_r = np.linalg.norm(food_cnt-food_centroid, axis=1)
    
    return food_cnt, food_r, food_centroid
#%%
def get_cnt_feats(skeletons, 
                  food_cnt, 
                  food_r, 
                  food_centroid,
                  is_debug = False):
    #%%
    partitions = ['head_base', 'tail_base', 'midbody']
    skel_avg = get_partition_transform(skeletons,
                                       func=np.mean,
                                       partitions=partitions
                                       )
    
    midbody_cc = skel_avg['midbody']
    
    def _get_food_ind(cc):
        rr = np.linalg.norm(cc - food_cnt, axis=1)
        ip = np.argmin(rr)
        rp = rr[ip]
        return ip, rp
    
    cnt_ind, dist_from_cnt = map(np.array, zip(*map(_get_food_ind, midbody_cc)))
    
    #find if the trajectory points are inside the closed polygon (outside will be negative)
    bbPath = mplPath.Path(food_cnt)
    outside = ~bbPath.contains_points(midbody_cc)
    dist_from_cnt[outside] = -dist_from_cnt[outside]
    
    ''' OLD
    #use the polar coordinates from the food centroid
    #to find if the worm is inside or outside the food
    #if worm radial position is larger tha the closest contour radial position
    #then the worm is outside
    midbody_r = np.linalg.norm(midbody_cc-food_centroid, axis=1)  
    outside = midbody_r>food_r[cnt_ind]
    dist_from_cnt[outside] = -dist_from_cnt[outside]
    '''
    
    get_unit_vec = lambda x : x/np.linalg.norm(x, axis=1)[:, np.newaxis]
    
    top = cnt_ind+1
    top[top>=food_cnt.shape[0]] -= food_cnt.shape[0] #fix any overflow index
    bot = cnt_ind-1 #it is not necessary to correct because we can use negative indexing
    
    food_u =  get_unit_vec(food_cnt[top]-food_cnt[bot])
    worm_u = get_unit_vec(skel_avg['head_base'] - skel_avg['tail_base'])
    dot_prod = np.sum(food_u*worm_u, axis=1)
    orientation_food_cnt = np.arccos(dot_prod)*180/np.pi
    #%%
    if is_debug:
        plt.figure(figsize=(12,12))
        
        plt.subplot(2,2,2)
        plt.plot(orientation_food_cnt)
        plt.title('Orientation respect to the food contour')
        
        plt.subplot(2,2,4)
        plt.plot(dist_from_cnt)
        plt.title('Distance from the food contour')
        
        plt.subplot(1,2,1)
        plt.plot(food_cnt[:,0], food_cnt[:,1])
        plt.plot(midbody_cc[:,0], midbody_cc[:,1], '.')
        plt.plot(food_cnt[cnt_ind,0], food_cnt[cnt_ind,1], 'r.')
        plt.axis('equal')
        
    #%%
    return orientation_food_cnt, dist_from_cnt, cnt_ind
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    import glob
    import os
    import fnmatch
    
    exts = ['']

    exts = ['*'+ext+'.hdf5' for ext in exts]
    
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_310517/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_160517/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_020617/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/Test_Food/MaskedVideos/FoodDilution_041116'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/Development_C1_170617/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/**/'
    #mask_dir = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/ATR_210417'
    mask_dir = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/**/'
    
    fnames = glob.glob(os.path.join(mask_dir, '*.hdf5'))
    fnames = [x for x in fnames if any(fnmatch.fnmatch(x, ext) for ext in exts)]
    
    #x.n_valid_skel/x.n_frames >= feat_filt_param['bad_seg_thresh']]
    for mask_video in fnames[1:]:
        skeletons_file = mask_video.replace('MaskedVideos','Results').replace('.hdf5', '_skeletons.hdf5')
        food_cnt, food_r, food_centroid = obtain_food_cnt(mask_video)
        
        good_traj_index, worm_index_type = getGoodTrajIndexes(skeletons_file)
        for iw, worm_index in enumerate(good_traj_index):
            worm = WormFromTableSimple(skeletons_file,
                                worm_index,
                                worm_index_type=worm_index_type)
            
            orientation_food_cnt, dist_from_cnt, cnt_ind = \
            get_cnt_feats(worm.skeleton, 
                  food_cnt, 
                  food_r, 
                  food_centroid,
                  is_debug = True)
            
            #%% 
            skeletons = worm.skeleton
            is_debug = True
        break
