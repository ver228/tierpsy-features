#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:54:00 2017

@author: ajaver
"""
import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import interp1d

from tierpsy_features.curvatures import _h_curvature_grad
from tierpsy_features.postures import get_length
from tierpsy_features.helper import DataPartition


path_curvature_columns = ['path_curvature_body', 
                          'path_curvature_tail', 
                          'path_curvature_midbody', 
                          'path_curvature_head'
                          ]

path_curvature_columns_aux = ['coord_x_body', 'coord_y_body', 
                              'coord_x_tail', 'coord_y_tail',
                              'coord_x_midbody', 'coord_y_midbody', 
                              'coord_x_head', 'coord_y_head'
                              ]

DFLT_ARGS = dict(
        path_step = 11,
        path_grad_window = 5,
        clip_val_body_lengths = 20
        )

#%%
def _h_path_curvature(skeletons, 
                      body_length = None,
                      partition_str = 'body', 
                      path_step = DFLT_ARGS['path_step'], 
                      path_grad_window = DFLT_ARGS['path_grad_window'],
                      clip_val_body_lengths = DFLT_ARGS['clip_val_body_lengths'],
                      _is_debug = False):
    
    if body_length is None:
        #caculate the length if it is not given
        body_length = np.nanmedian(get_length(skeletons))
    clip_val = clip_val_body_lengths/body_length
    
    p_obj = DataPartition(n_segments=skeletons.shape[1])
    body_coords = p_obj.apply(skeletons, partition_str, func=np.mean)
    
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
    
    fx = interp1d(lengths, xx_i)
    fy = interp1d(lengths, yy_i)
    ft = interp1d(lengths, tt)
    
    sub_lengths = np.arange(lengths[0], lengths[-1], path_step)
    
    #there is not enough data to calculate the curvature
    if len(sub_lengths) <= 4*path_grad_window:
        return np.full(skeletons.shape[0], np.nan), body_coords
    
    xs = fx(sub_lengths)
    ys = fy(sub_lengths)
    ts = ft(sub_lengths)
    
    curve = np.vstack((xs, ys)).T
    curvature_r = _h_curvature_grad(curve, 
                                    points_window = path_grad_window, 
                                    axis=0,
                                    is_nan_border=False)
    
    #clip values to remove regions with extremely large curvatures (typically short reversars)
    curvature_r = np.clip(curvature_r, -clip_val,clip_val)
    
    ts_i = np.hstack((-1, ts, tt[-1] + 1))
    c_i = np.hstack((curvature_r[0], curvature_r, curvature_r[-1]))
    curvature_t = interp1d(ts_i, c_i)(tt)
    
    if _is_debug:
        import matplotlib.pylab as plt
        from matplotlib.collections import LineCollection
        #path_curvature[np.isnan(worm_features['speed'])] = np.nan
        #path_curvature = np.clip(curvature_t, -0.02, 0.02)
        path_curvature = curvature_t
        
        curv_range = (np.nanmin(path_curvature), np.nanmax(path_curvature))
        
        points = body_coords.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, 
                            cmap = plt.get_cmap('plasma'),
                            norm = plt.Normalize(*curv_range))
        lc.set_array(path_curvature)
        lc.set_linewidth(2)
    
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1,2,1)
        plt.gca().add_collection(lc)
    
        plt.xlim(3000, 11000)
        plt.ylim(3000, 11000)
        plt.axis('equal')
        
        plt.subplot(1,2,2)
        plt.plot(path_curvature)
    
    return curvature_t, body_coords

def get_path_curvatures(skeletons, **argkws):
    path_curvatures = []    
    path_coords = []
    
    body_length = np.nanmedian(get_length(skeletons))
    
    for partition_str in ['body', 'tail', 'midbody', 'head']:
        
        
        path_curv, coords = \
            _h_path_curvature(skeletons, 
                              body_length,
                              partition_str = partition_str, 
                              **argkws
                              )
        
        path_curvatures.append(('path_curvature_' + partition_str, path_curv))
        
        path_coords.append(('coord_x_' + partition_str, coords[...,0]))
        path_coords.append(('coord_y_' + partition_str, coords[...,1]))
        
    
    cols, dat = zip(*path_curvatures)
    path_curvatures_df = pd.DataFrame(np.array(dat).T, columns=cols)
    
    cols, dat = zip(*path_coords)
    path_coords_df = pd.DataFrame(np.array(dat).T, columns=cols)
    return path_curvatures_df, path_coords_df

def _test_plot_cnts_maps(ventral_contour, dorsal_contour):
    import matplotlib.pylab as plt
    pix2microns = 10
    
    x_min = np.nanmin(ventral_contour[:, :, 0])
    x_max = np.nanmax(ventral_contour[:, :, 0])
    
    y_min = np.nanmin(dorsal_contour[:, :, 1])
    y_max = np.nanmax(dorsal_contour[:, :, 1])
    
    
    rx = int(round((x_max - x_min)/pix2microns))
    ry = int(round((y_max - y_min)/pix2microns))
    
    size_counts = (rx + 1, ry + 1)
    
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
        
        for ii, cnt in enumerate(cnts):
            if np.any(np.isnan(cnt)):
                continue
            cc = np.zeros(size_counts, np.float32)
            cc = cv2.drawContours(cc, [cnt[:, None, :].astype(np.int)], contourIdx=-1, thickness=-1, color=1)
            counts += cc
        
        plt.figure()
        plt.imshow(counts, interpolation='none')
        plt.title(part)
        
        all_cnts[part] = counts
    
        print(part)
      
    plt.figure(); 
    plt.imshow(np.clip(all_cnts['head']-all_cnts['body'], 0, 10000), interpolation='none')

#%%
if __name__ == '__main__':
    import os
    import tables
    
    #%%
    #_test_plot_cnts_maps(ventral_contour, dorsal_contour)
    base_dir = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm'
    skeletons_file = os.path.join(base_dir, 'MY23_worms5_food1-10_Set4_Pos5_Ch4_29062017_140148_skeletons.hdf5')
    features_file = skeletons_file.replace('_skeletons.hdf5', '_featuresN.hdf5')
    
    #features_file = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm/MY16_worms5_food1-10_Set5_Pos4_Ch1_02062017_131004_featuresN.hdf5'
    features_file = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm/170817_matdeve_exp7co1_12_Set0_Pos0_Ch1_17082017_140001_featuresN.hdf5'
    
    with pd.HDFStore(features_file, 'r') as fid:
        blob_features = fid['/blob_features']
        trajectories_data = fid['/trajectories_data']
        timeseries_features = fid['/timeseries_features']
        aux_features = fid['/aux_features']
        
        fps = fid.get_storer('/trajectories_data').attrs['fps']
        
    if False:
        good = trajectories_data['skeleton_id']>=0
        trajectories_data = trajectories_data[good]
        blob_features = blob_features[good]
        
        trajectories_data_g = trajectories_data.groupby('worm_index_joined')
        
        
        for worm_index in trajectories_data_g.groups.keys():
            worm_index = 4#695
            worm_data = trajectories_data_g.get_group(worm_index)
            skel_id = worm_data['skeleton_id'].values 
            with tables.File(features_file, 'r') as fid:
                skeletons = fid.get_node('/coordinates/skeletons')[skel_id, :, :]
            worm_features = timeseries_features.loc[skel_id]
            
            path_curvatures_df, path_coords_df = get_path_curvatures(skeletons, _is_debug=True)
            break
    #%%
    import matplotlib.pylab as plt
    
    cols = [x for x in aux_features if 'coord_' in x]
    path_coords_df = aux_features[cols]
    
    bin_size_b_length = 0.25
    
    body_length = timeseries_features['length'].median()
    bin_size_microns = bin_size_b_length*body_length
    
    bin_vals = ((path_coords_df - path_coords_df.mean())/bin_size_microns).round()
    bin_vals = bin_vals.fillna(method='ffill').astype(np.int)
    
    #%%
    bin_counts = {}
    for b_part in set(x.rpartition('_')[-1] for x in bin_vals.columns):
        dat = bin_vals[['coord_x_' + b_part,'coord_y_' + b_part]]
        dat.columns = ['X', 'Y']
        
        gg = dat.groupby(["X", "Y"])
        cc = gg.size().reset_index(name="Counts")
        bin_counts[b_part] = cc
        #cc = pd.crosstab(dat['X'], dat['Y'])
        #[gg.groups.keys()]
    #%%  
        
    for k, val in map_counts.items():
        plt.figure()
        plt.imshow(val.values, interpolation='none')
        plt.title(k)
    #%%
    plt.figure()
    dd = map_counts['head']-map_counts['body']
    dd[np.isnan(dd) | (dd<0)] = 0
    plt.imshow(dd, interpolation='none')
    
#%% 
    b_counts = cc.values[cc>0]/fps
    t_med = np.median(b_counts)
    t_max = np.max(b_counts)
    b_area = b_counts.size*(bin_size_microns**2) #tot area
    #%%
    dd = set([tuple(x) for x in dat[['X', 'Y']].values])