#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:49:09 2017

@author: ajaver
"""

from sklearn.linear_model import Lasso
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:05:32 2017

@author: ajaver
"""
import pandas as pd
import pymysql
import numpy as np


bad_feats = ['head_orientation_forward_pos', 'tail_to_head_orientation_abs', 
             'tail_orientation_paused_neg', 'tail_orientation_backward_neg', 
             'tail_orientation_neg', 'tail_to_head_orientation_paused_abs', 
             'head_orientation_forward_abs', 'tail_orientation_abs', 
             'tail_orientation_backward_abs', 'tail_to_head_orientation_backward_abs', 
             'head_orientation_backward_abs', 'head_orientation_backward_pos', 
             'head_orientation_forward_neg', 'head_orientation_abs', 
             'tail_orientation_paused_abs', 'tail_to_head_orientation_neg', 
             'tail_to_head_orientation_forward_pos', 'tail_to_head_orientation_backward_neg', 
             'path_curvature', 'tail_to_head_orientation_forward_abs', 
             'tail_to_head_orientation_paused_neg', 'head_orientation_neg', 
             'tail_orientation_forward_abs', 'tail_orientation_forward_neg', 
             'head_orientation_pos', 'tail_to_head_orientation_pos', 
             'tail_to_head_orientation_backward_pos', 'tail_orientation_backward_pos', 
             'head_orientation_backward_neg', 'tail_to_head_orientation_backward', 
             'head_orientation_paused_abs', 'head_orientation_paused_neg', 
             'path_curvature_backward', 'tail_to_head_orientation', 
             'tail_orientation', 'tail_to_head_orientation_paused_pos', 
             'head_orientation_backward', 'tail_orientation_backward', 
             'tail_to_head_orientation_paused', 'tail_orientation_pos', 
             'head_orientation_forward', 'tail_to_head_orientation_forward_neg', 
             'head_orientation_paused_pos', 'path_curvature_forward', 
             'head_orientation', 'tail_to_head_orientation_forward', 
             'tail_orientation_forward_pos', 'tail_orientation_forward', 
             'tail_orientation_paused_pos', 'head_orientation_paused', 
             'path_curvature_paused', 'tail_orientation_paused'
             ]
redundant_feats = ['omega_turns_time_abs', 'omega_turns_time_pos'] #this are the same as omega_turns_time

almost_nan_feats = ['midbody_speed_forward_neg', 'midbody_speed_backward_pos',
       'head_crawling_amplitude_paused', 'head_crawling_amplitude_paused_abs',
       'head_crawling_amplitude_paused_neg',
       'head_crawling_amplitude_paused_pos',
       'midbody_crawling_amplitude_paused',
       'midbody_crawling_amplitude_paused_abs',
       'midbody_crawling_amplitude_paused_neg',
       'midbody_crawling_amplitude_paused_pos',
       'tail_crawling_amplitude_paused', 'tail_crawling_amplitude_paused_abs',
       'tail_crawling_amplitude_paused_neg',
       'tail_crawling_amplitude_paused_pos', 'head_crawling_frequency_paused',
       'head_crawling_frequency_paused_abs',
       'head_crawling_frequency_paused_neg',
       'head_crawling_frequency_paused_pos',
       'midbody_crawling_frequency_paused',
       'midbody_crawling_frequency_paused_abs',
       'midbody_crawling_frequency_paused_neg',
       'midbody_crawling_frequency_paused_pos',
       'tail_crawling_frequency_paused', 'tail_crawling_frequency_paused_abs',
       'tail_crawling_frequency_paused_neg',
       'tail_crawling_frequency_paused_pos', 'coils_time', 'inter_coils_time',
       'inter_coils_distance', 'coils_frequency', 'coils_time_ratio',
       'omega_turns_time_neg', 'inter_omega_turns_time',
       'inter_omega_turns_time_abs', 'inter_omega_turns_time_neg',
       'inter_omega_turns_time_pos', 'inter_omega_turns_distance',
       'inter_omega_turns_distance_abs', 'inter_omega_turns_distance_neg',
       'inter_omega_turns_distance_pos', 'upsilon_turns_time',
       'upsilon_turns_time_abs', 'upsilon_turns_time_neg',
       'upsilon_turns_time_pos', 'inter_upsilon_turns_time',
       'inter_upsilon_turns_time_abs', 'inter_upsilon_turns_time_neg',
       'inter_upsilon_turns_time_pos', 'inter_upsilon_turns_distance',
       'inter_upsilon_turns_distance_abs', 'inter_upsilon_turns_distance_neg',
       'inter_upsilon_turns_distance_pos', 'upsilon_turns_frequency',
       'upsilon_turns_time_ratio'
       ]


#%%
if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', database='single_worm_db')
    
    sql = '''
    SELECT e.strain, e.date, feat_m.* 
    FROM experiments_valid AS e
    JOIN features_means AS feat_m ON e.id = feat_m.experiment_id
    WHERE total_time < 905
    AND total_time > 295
    AND n_valid_skeletons > 120*fps
    '''
    
    df = pd.read_sql(sql, con=conn)
    #%%
    index_feats = ['strain', 'experiment_id', 'worm_index', 
                   'n_frames', 'n_valid_skel', 'first_frame', 'date']
    
    feats2remove = index_feats + bad_feats + almost_nan_feats + redundant_feats
    feats2check = [x for x in df.columns if not x in feats2remove]
    
    df_v = df[feats2check]
    df_v = df_v[(df_v.T.isnull().mean()<0.1).values]
    df_v = df_v.fillna(df_v.mean())
    #%%
    M = df_v.mean()
    S = df_v.std()
    df_n = (df_v - M)/S
    #%%
    #U, s, V = np.linalg.svd(df_n) #singular value descomposition useful to find linear combinations if s is small
    
    #%%
    from sklearn.decomposition import PCA
    
    pca = PCA()
    pca.fit(df_n)
    #%%
    g = sns.clustermap(corr_mat.abs(), method='single')
    #%%
    corr_mat = df[feats2check].corr()
    #%%
    gg = [corr_mat.index[corr_mat[col].abs()>0.99] for col in corr_mat]
    gg = [list(x) for x in gg if len(x) > 1]
    
    gg_reduced = []
    for x in gg:
        if not x in gg_reduced:
            gg_reduced.append(x)
    #%%
    for x in gg_reduced:
        print(sorted(x))
    #%%
['length', 'length_backward', 'length_forward', 'length_paused']
['area_length_ratio', 'midbody_width', 'midbody_width_backward', 'midbody_width_paused']
['area_length_ratio_forward', 'midbody_width_forward']
['midbody_width', 'midbody_width_paused']
['area_length_ratio_backward', 'midbody_width', 'midbody_width_backward']
['area', 'area_backward', 'area_forward', 'area_paused']
['area_length_ratio', 'area_length_ratio_backward', 'area_length_ratio_forward', 'area_length_ratio_paused', 'midbody_width']
['area_length_ratio', 'area_length_ratio_forward', 'midbody_width_forward']
['area_length_ratio', 'area_length_ratio_paused']
['area_length_ratio', 'area_length_ratio_backward', 'midbody_width_backward']
['head_speed', 'head_tip_speed', 'midbody_speed', 'tail_speed', 'tail_tip_speed']
['head_speed_abs', 'head_tip_speed_abs', 'head_tip_speed_pos']
['head_speed_neg', 'head_tip_speed_neg']
['head_speed_abs', 'head_speed_pos', 'head_tip_speed_abs', 'head_tip_speed_pos']
['head_speed_forward', 'head_tip_speed_forward']
['head_tip_speed_forward_abs', 'head_tip_speed_forward_pos']
['head_speed_abs', 'head_speed_pos', 'head_tip_speed_abs', 'head_tip_speed_pos', 'midbody_speed_abs', 'midbody_speed_pos', 'tail_speed_abs', 'tail_speed_pos', 'tail_tip_speed_abs']
['head_speed_abs', 'head_speed_pos', 'head_tip_speed_pos', 'midbody_speed_abs', 'midbody_speed_pos', 'tail_speed_abs', 'tail_speed_pos', 'tail_tip_speed_pos']
['head_speed_forward_abs', 'head_speed_forward_pos']
['head_speed_backward_abs', 'midbody_speed_backward', 'midbody_speed_backward_abs', 'midbody_speed_backward_neg']
['head_speed_abs', 'head_speed_pos', 'midbody_speed_abs', 'midbody_speed_pos', 'tail_speed_abs', 'tail_speed_pos', 'tail_tip_speed_abs', 'tail_tip_speed_pos']
['midbody_speed_neg', 'tail_speed_neg', 'tail_tip_speed_neg']
['midbody_speed_forward', 'midbody_speed_forward_abs', 'midbody_speed_forward_pos', 'tail_speed_forward_abs', 'tail_speed_forward_pos']
['midbody_speed_forward', 'midbody_speed_forward_abs', 'midbody_speed_forward_pos', 'tail_speed_forward_abs', 'tail_speed_forward_pos', 'tail_tip_speed_forward_abs']
['head_speed_backward_abs', 'midbody_speed_backward', 'midbody_speed_backward_abs', 'midbody_speed_backward_neg', 'tail_speed_backward_abs', 'tail_speed_backward_neg']
['tail_speed_forward', 'tail_tip_speed_forward']
['midbody_speed_forward', 'midbody_speed_forward_abs', 'midbody_speed_forward_pos', 'tail_speed_forward_abs', 'tail_speed_forward_pos', 'tail_tip_speed_forward_abs', 'tail_tip_speed_forward_pos']
['midbody_speed_backward', 'midbody_speed_backward_abs', 'midbody_speed_backward_neg', 'tail_speed_backward_abs', 'tail_speed_backward_neg']
['midbody_speed_backward', 'midbody_speed_backward_abs', 'midbody_speed_backward_neg', 'tail_speed_backward_abs', 'tail_speed_backward_neg', 'tail_tip_speed_backward_neg']
['head_speed_abs', 'midbody_speed_abs', 'midbody_speed_pos', 'tail_speed_abs', 'tail_speed_pos', 'tail_tip_speed_abs', 'tail_tip_speed_pos']
['head_speed_pos', 'midbody_speed_abs', 'midbody_speed_pos', 'tail_speed_abs', 'tail_speed_pos', 'tail_tip_speed_abs', 'tail_tip_speed_pos']
['midbody_speed_forward_abs', 'midbody_speed_forward_pos', 'tail_speed_forward_abs', 'tail_speed_forward_pos', 'tail_tip_speed_forward_abs', 'tail_tip_speed_forward_pos']
['tail_speed_forward_abs', 'tail_speed_forward_pos', 'tail_tip_speed_forward_abs', 'tail_tip_speed_forward_pos']
['tail_speed_backward_neg', 'tail_tip_speed_backward_neg']
['head_tip_motion_direction_abs', 'head_tip_motion_direction_neg', 'head_tip_motion_direction_pos']
['head_tip_motion_direction_abs', 'head_tip_motion_direction_neg']
['head_tip_motion_direction_abs', 'head_tip_motion_direction_pos']
['head_motion_direction_abs', 'head_motion_direction_neg', 'head_motion_direction_pos']
['head_motion_direction_abs', 'head_motion_direction_neg']
['head_motion_direction_abs', 'head_motion_direction_pos']
['midbody_motion_direction_abs', 'midbody_motion_direction_neg', 'midbody_motion_direction_pos']
['tail_motion_direction_abs', 'tail_motion_direction_neg', 'tail_motion_direction_pos']
['tail_motion_direction_abs', 'tail_motion_direction_neg']
['tail_motion_direction_abs', 'tail_motion_direction_pos']
['tail_tip_motion_direction_abs', 'tail_tip_motion_direction_neg', 'tail_tip_motion_direction_pos']
['foraging_speed_abs', 'foraging_speed_neg', 'foraging_speed_pos']
['foraging_speed_abs', 'foraging_speed_neg']
['foraging_speed_abs', 'foraging_speed_pos']
['midbody_crawling_frequency_forward_abs', 'midbody_crawling_frequency_forward_neg']
['path_curvature_abs', 'path_curvature_neg', 'path_curvature_pos']
['path_curvature_abs', 'path_curvature_neg']
['path_curvature_abs', 'path_curvature_pos']
['path_curvature_paused_abs', 'path_curvature_paused_neg', 'path_curvature_paused_pos']
['path_curvature_paused_abs', 'path_curvature_paused_neg']
['path_curvature_paused_abs', 'path_curvature_paused_pos']
['omega_turns_time', 'omega_turns_time_abs', 'omega_turns_time_pos']
    
    #%%
#    corr_mat = df[feats2check].corr()
#    np.fill_diagonal(corr_mat.values, np.nan)
#    
    #%%
    #corr_mat_spearman = df[feats2check].corr('spearman')
    
#    #%%
#    import matplotlib.pylab as plt
#    import seaborn as sns; sns.set(color_codes=True)
#    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#    
#    #%%
#    #g = sns.clustermap(corr_mat.abs(), method='single')
#    g = sns.clustermap(corr_mat_spearman.abs(), method='single', )
#    Z = g.dendrogram_col.calculated_linkage
#    
#    
#    #%%
#    #k = 12
#    #clusters = fcluster(Z, k, criterion='maxclust')
#    max_d = 1
#    clusters = fcluster(Z, max_d, criterion='distance')
#    #%%
#    cluster_sizes = [(ii, v) for ii, v in enumerate(np.bincount(clusters)) if v > 0]
#    #%%
#    for kk in dd:
#        g_cols = corr_mat.columns[clusters == kk]
#        
#        print(kk, len(g_cols))
#        for cols in g_cols:
#            print(cols)
#    #%%
#    no_cluster = corr_mat.columns[[x for x, n in cluster_sizes if n >= 2]]
#    
#    #%%
#    
    #%%
    #
#    #%%
#    Z = linkage(corr_mat, 'single')
#    out = dendrogram(Z)
#    
#    
#    #%%
#    
#    Z = g.dendrogram_col.calculated_linkage
#    out = dendrogram(Z,  truncate_mode='lastp',  p=30)
    
    