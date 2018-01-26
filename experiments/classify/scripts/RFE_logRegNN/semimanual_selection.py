#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018S

@author: ajaver
S
Modified from: 
    https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/2_logistic_regression.py
"""

import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

import multiprocessing as mp


#%%
import torch
from torch.autograd import Variable
from trainer import TrainerSimpleNet, SimpleNet
from reader import read_feats

#%%    
def simple_fit(data_in):
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test) = fold_data
    (cuda_id, n_epochs, batch_size) = fold_param
    
    
    n_classes = int(y_train.max() + 1)
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long() 
    
    input_v = Variable(x_test.cuda(cuda_id), requires_grad=False)
    target_v = Variable(y_test.cuda(cuda_id), requires_grad=False)
    
    input_train = x_train
    target_train = y_train
    
    n_features = input_v.size(1)
    trainer = TrainerSimpleNet(n_classes, n_features, n_epochs, batch_size, cuda_id)
    trainer.fit(input_train, target_train)
    res = trainer.evaluate(input_v, target_v)
    
    print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*res))
    
    return (fold_id, res)
#%%
if __name__ == "__main__":
    feat_data, col2ignore_r = read_feats()
    
#%%
   
#    core_sets = [
#            ['length_50th',
#        'curvature_head_IQR',
#        'quirkiness_50th',
#        'speed_90th',
#        'motion_mode_backward_fraction'],
#             
#              ['curvature_head_abs_90th',
# 'width_midbody_norm_50th',
# 'length_50th',
# 'curvature_hips_abs_90th',
# 'motion_mode_backward_fraction'],
#               
#               ['curvature_head_abs_90th',
# 'width_midbody_norm_50th',
# 'length_50th',
# 'curvature_midbody_abs_90th',
# 'motion_mode_backward_fraction'],
#               
#                ['curvature_head_abs_IQR',
# 'width_midbody_norm_50th',
# 'curvature_head_abs_90th',
# 'curvature_hips_abs_90th',
# 'motion_mode_backward_fraction'],
#                 
#            ['curvature_head_abs_90th',
# 'width_midbody_norm_90th',
# 'd_curvature_hips_90th',
# 'd_curvature_head_90th',
# 'motion_mode_backward_fraction'],
#        
#   ['curvature_head_IQR',
# 'width_midbody_norm_90th',
# 'd_curvature_hips_90th',
# 'd_curvature_head_90th',
# 'motion_mode_backward_fraction'],
#    ]
   
#rfe_sets = [
#            ['curvature_head_norm_abs_90th',
# 'width_midbody_norm_90th',
# 'length_90th',
# 'd_curvature_hips_10th',
# 'motion_mode_backward_fraction'],
#             
#    ['curvature_head_norm_abs_IQR',
# 'd_curvature_head_90th',
# 'width_midbody_norm_10th',
# 'd_curvature_hips_10th',
# 'motion_mode_backward_fraction'],
#     
#     ['curvature_head_norm_abs_90th',
#  'd_eigen_projection_6_IQR',
#  'width_head_base_norm_10th',
#  'blob_compactness_50th',
#  'motion_mode_paused_fraction'
#  ]
# ]
    #%%
    core_sets = [
        ['length_50th',
         'width_midbody_50th',
        'curvature_head_abs_50th',
        'curvature_midbody_abs_50th',
        'speed_90th',
        'speed_10th',
        'motion_mode_backward_fraction',
        'motion_mode_forward_fraction'
        ],
        
        ['length_50th',
         'width_midbody_norm_50th',
         'curvature_hips_abs_90th',
         'curvature_head_abs_90th',
         'motion_mode_backward_fraction',
         'motion_mode_forward_frequency',
         'd_curvature_hips_90th',
         'd_curvature_head_90th',
         ]
        ]
    rfe_sets = [['width_head_base_norm_50th',
  'blob_solidity_w_forward_50th',
  'path_coverage_tail',
  'd_curvature_head_w_forward_IQR',
  'd_eigen_projection_6_IQR',
  'curvature_head_w_forward_IQR',
  'curvature_head_abs_90th',
  'motion_mode_paused_fraction'],
 ['path_coverage_tail',
  'blob_quirkiness_w_forward_50th',
  'd_curvature_head_w_forward_IQR',
  'width_head_base_w_forward_10th',
  'curvature_head_w_forward_IQR',
  'curvature_head_norm_abs_90th',
  'blob_solidity_w_forward_50th',
  'motion_mode_paused_fraction'],
 ['curvature_midbody_norm_abs_50th',
  'width_tail_base_w_backward_10th',
  'd_curvature_head_w_forward_IQR',
  'curvature_head_norm_abs_90th',
  'd_eigen_projection_6_IQR',
  'width_head_base_norm_10th',
  'blob_compactness_50th',
  'motion_mode_paused_fraction'],
 ['d_curvature_head_w_forward_IQR',
  'curvature_head_norm_abs_90th',
  'd_eigen_projection_6_IQR',
  'motion_mode_paused_fraction',
  'motion_mode_paused_frequency',
  'blob_quirkiness_50th',
  'width_head_base_w_forward_10th',
  'length_w_backward_10th'],
 ['path_coverage_tail',
  'curvature_head_norm_IQR',
  'd_eigen_projection_6_w_forward_IQR',
  'motion_mode_paused_fraction',
  'motion_mode_paused_frequency',
  'quirkiness_50th',
  'length_w_backward_90th',
  'width_head_base_w_forward_10th']]
#        #%%
#    rfe_sets = [['d_relative_radial_velocity_tail_tip_IQR',
#  'curvature_head_norm_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'd_curvature_hips_10th',
#  'd_quirkiness_IQR',
#  'curvature_midbody_norm_IQR',
#  'd_length_IQR',
#  'motion_mode_paused_fraction'],
# ['d_quirkiness_IQR',
#  'path_coverage_tail',
#  'curvature_midbody_norm_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'width_midbody_norm_90th',
#  'length_10th',
#  'd_curvature_hips_10th',
#  'motion_mode_paused_fraction'],
# ['d_curvature_neck_IQR',
#  'quirkiness_90th',
#  'motion_mode_paused_frequency',
#  'curvature_head_norm_abs_IQR',
#  'd_curvature_head_90th',
#  'width_midbody_norm_10th',
#  'd_curvature_hips_10th',
#  'motion_mode_backward_fraction'],
# ['d_area_IQR',
#  'width_midbody_norm_10th',
#  'curvature_neck_norm_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'curvature_head_norm_IQR',
#  'motion_mode_backward_fraction',
#  'relative_speed_midbody_norm_abs_90th',
#  'motion_mode_forward_fraction'],
# ['d_quirkiness_IQR',
#  'curvature_head_norm_abs_50th',
#  'relative_speed_midbody_norm_abs_90th',
#  'd_major_axis_10th',
#  'd_curvature_hips_10th',
#  'motion_mode_backward_fraction',
#  'curvature_midbody_norm_abs_90th',
#  'curvature_tail_norm_abs_90th'],
# ['d_relative_radial_velocity_tail_tip_IQR',
#  'curvature_tail_norm_abs_90th',
#  'curvature_head_norm_IQR',
#  'd_relative_radial_velocity_head_tip_50th',
#  'd_quirkiness_IQR',
#  'relative_speed_midbody_norm_abs_90th',
#  'curvature_hips_norm_abs_90th',
#  'motion_mode_paused_fraction'],
# ['d_area_IQR',
#  'width_midbody_10th',
#  'path_coverage_tail',
#  'curvature_neck_norm_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'width_midbody_norm_10th',
#  'd_curvature_hips_10th',
#  'motion_mode_paused_fraction'],
# ['d_quirkiness_IQR',
#  'motion_mode_forward_frequency',
#  'path_coverage_tail',
#  'curvature_head_norm_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'width_midbody_10th',
#  'width_midbody_norm_10th',
#  'motion_mode_paused_fraction'],
# ['d_curvature_neck_IQR',
#  'area_10th',
#  'motion_mode_backward_frequency',
#  'relative_radial_velocity_head_tip_norm_50th',
#  'd_relative_speed_midbody_10th',
#  'length_90th',
#  'curvature_tail_norm_abs_90th',
#  'motion_mode_paused_frequency'],
# ['d_curvature_hips_10th',
#  'minor_axis_50th',
#  'path_coverage_tail',
#  'curvature_tail_norm_abs_90th',
#  'd_length_IQR',
#  'd_quirkiness_IQR',
#  'd_curvature_head_IQR',
#  'motion_mode_paused_fraction'],
# ['d_length_IQR',
#  'quirkiness_90th',
#  'path_coverage_tail',
#  'curvature_tail_norm_abs_90th',
#  'curvature_head_norm_IQR',
#  'length_90th',
#  'd_curvature_hips_10th',
#  'motion_mode_paused_fraction'],
# ['d_length_IQR',
#  'area_10th',
#  'motion_mode_paused_frequency',
#  'relative_radial_velocity_head_tip_norm_50th',
#  'curvature_tail_norm_abs_90th',
#  'motion_mode_backward_fraction',
#  'd_curvature_hips_10th',
#  'quirkiness_90th'],
# ['d_quirkiness_IQR',
#  'path_coverage_tail',
#  'curvature_midbody_norm_abs_50th',
#  'relative_radial_velocity_head_tip_norm_50th',
#  'd_curvature_neck_IQR',
#  'length_90th',
#  'width_midbody_norm_90th',
#  'quirkiness_50th'],
# ['d_length_IQR',
#  'quirkiness_50th',
#  'motion_mode_paused_frequency',
#  'curvature_head_norm_abs_90th',
#  'd_relative_speed_midbody_10th',
#  'curvature_neck_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'd_curvature_hips_10th'],
# ['d_area_IQR',
#  'major_axis_10th',
#  'motion_mode_backward_fraction',
#  'relative_radial_velocity_head_tip_norm_50th',
#  'curvature_tail_norm_abs_90th',
#  'd_curvature_hips_10th',
#  'd_curvature_head_90th',
#  'motion_mode_paused_fraction'],
# ['d_quirkiness_IQR',
#  'width_midbody_10th',
#  'width_midbody_norm_90th',
#  'd_curvature_head_90th',
#  'd_length_IQR',
#  'major_axis_10th',
#  'd_curvature_hips_10th',
#  'motion_mode_paused_fraction'],
# ['d_length_IQR',
#  'motion_mode_paused_frequency',
#  'curvature_midbody_norm_abs_90th',
#  'width_midbody_norm_90th',
#  'd_relative_speed_midbody_10th',
#  'length_10th',
#  'd_curvature_hips_10th',
#  'motion_mode_backward_fraction'],
# ['d_curvature_neck_IQR',
#  'motion_mode_forward_frequency',
#  'path_density_head_95th',
#  'relative_radial_velocity_head_tip_norm_50th',
#  'relative_radial_velocity_head_tip_norm_90th',
#  'motion_mode_paused_frequency',
#  'curvature_midbody_norm_IQR',
#  'curvature_tail_norm_abs_90th'],
# ['d_curvature_head_90th',
#  'motion_mode_paused_frequency',
#  'relative_radial_velocity_head_tip_norm_90th',
#  'd_relative_speed_midbody_10th',
#  'd_curvature_hips_10th',
#  'width_midbody_10th',
#  'd_quirkiness_10th',
#  'curvature_tail_norm_abs_90th'],
# ['d_length_IQR',
#  'motion_mode_paused_frequency',
#  'curvature_midbody_norm_abs_50th',
#  'curvature_head_norm_abs_90th',
#  'width_midbody_norm_90th',
#  'length_90th',
#  'd_curvature_hips_10th',
#  'motion_mode_backward_fraction'],
# ['d_length_IQR',
#  'relative_radial_velocity_head_tip_norm_50th',
#  'curvature_head_norm_abs_90th',
#  'curvature_tail_norm_abs_90th',
#  'd_quirkiness_IQR',
#  'width_midbody_10th',
#  'd_curvature_midbody_IQR',
#  'motion_mode_backward_fraction'],
# ['d_quirkiness_IQR',
#  'motion_mode_backward_frequency',
#  'curvature_head_norm_IQR',
#  'd_relative_speed_midbody_10th',
#  'd_curvature_hips_10th',
#  'length_90th',
#  'motion_mode_forward_frequency',
#  'motion_mode_paused_fraction'],
# ['d_area_IQR',
#  'motion_mode_backward_frequency',
#  'motion_mode_forward_frequency',
#  'curvature_tail_norm_abs_90th',
#  'd_length_IQR',
#  'length_90th',
#  'd_curvature_hips_10th',
#  'motion_mode_paused_fraction'],
# ['d_quirkiness_IQR',
#  'width_midbody_10th',
#  'curvature_head_norm_abs_50th',
#  'curvature_tail_norm_abs_90th',
#  'd_curvature_hips_10th',
#  'motion_mode_backward_fraction',
#  'width_midbody_norm_10th',
#  'motion_mode_forward_fraction'],
# ['d_area_IQR',
#  'minor_axis_50th',
#  'width_midbody_norm_90th',
#  'd_relative_speed_midbody_10th',
#  'd_curvature_hips_10th',
#  'quirkiness_50th',
#  'curvature_head_norm_IQR',
#  'motion_mode_paused_fraction']]
    
#    rfe_sets = [['d_area_w_forward_IQR',
#  'width_head_base_50th',
#  'curvature_midbody_w_forward_IQR',
#  'blob_solidity_10th',
#  'd_curvature_head_w_forward_IQR',
#  'curvature_head_norm_abs_90th',
#  'eigen_projection_7_IQR',
#  'motion_mode_paused_fraction'],
# ['d_width_head_base_IQR',
#  'curvature_neck_abs_90th',
#  'width_head_base_norm_10th',
#  'curvature_head_norm_abs_90th',
#  'd_curvature_head_w_forward_IQR',
#  'width_midbody_norm_10th',
#  'd_eigen_projection_6_IQR',
#  'motion_mode_paused_fraction'],
# ['d_eigen_projection_6_w_forward_IQR',
#  'motion_mode_backward_fraction',
#  'd_relative_speed_midbody_10th',
#  'd_curvature_neck_w_backward_10th',
#  'd_curvature_head_w_forward_IQR',
#  'width_head_base_w_forward_10th',
#  'width_head_base_norm_50th',
#  'blob_solidity_50th'],
# ['d_blob_solidity_w_forward_90th',
#  'relative_radial_velocity_tail_tip_w_backward_10th',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_forward_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'd_curvature_midbody_w_backward_90th',
#  'd_eigen_projection_6_w_forward_IQR',
#  'motion_mode_backward_fraction'],
# ['d_width_head_base_IQR',
#  'width_tail_base_w_backward_10th',
#  'curvature_tail_w_backward_IQR',
#  'curvature_head_norm_abs_90th',
#  'd_curvature_head_w_forward_IQR',
#  'blob_compactness_50th',
#  'd_eigen_projection_6_IQR',
#  'motion_mode_paused_fraction'],
# ['d_width_tail_base_IQR',
#  'width_head_base_norm_10th',
#  'curvature_head_norm_abs_90th',
#  'blob_quirkiness_50th',
#  'd_curvature_head_w_forward_IQR',
#  'curvature_midbody_norm_abs_90th',
#  'd_eigen_projection_6_w_forward_IQR',
#  'motion_mode_paused_fraction'],
# ['d_eigen_projection_6_IQR',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_forward_IQR',
#  'd_relative_speed_midbody_10th',
#  'd_curvature_head_w_forward_IQR',
#  'width_midbody_norm_10th',
#  'curvature_head_norm_abs_90th',
#  'motion_mode_paused_fraction'],
# ['d_curvature_neck_w_forward_IQR',
#  'motion_mode_paused_frequency',
#  'relative_radial_velocity_tail_tip_w_backward_10th',
#  'width_midbody_w_forward_10th',
#  'blob_compactness_50th',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_forward_IQR',
#  'curvature_head_norm_abs_50th'],
# ['d_width_head_base_IQR',
#  'curvature_head_w_backward_IQR',
#  'curvature_head_w_forward_IQR',
#  'blob_perimeter_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'width_midbody_norm_10th',
#  'd_eigen_projection_6_IQR',
#  'motion_mode_paused_fraction'],
# ['d_eigen_projection_6_w_forward_IQR',
#  'motion_mode_backward_fraction',
#  'path_coverage_tail',
#  'd_curvature_neck_w_backward_10th',
#  'd_curvature_head_w_forward_IQR',
#  'width_head_base_w_forward_10th',
#  'blob_quirkiness_50th',
#  'major_axis_90th'],
# ['d_curvature_neck_w_forward_IQR',
#  'path_coverage_tail',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_forward_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'blob_solidity_50th',
#  'eigen_projection_6_IQR',
#  'motion_mode_paused_fraction'],
# ['d_area_w_backward_IQR',
#  'width_head_base_w_forward_10th',
#  'curvature_head_norm_abs_90th',
#  'd_relative_radial_velocity_head_tip_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'length_90th',
#  'curvature_head_w_forward_IQR',
#  'motion_mode_paused_fraction'],
# ['d_quirkiness_w_forward_IQR',
#  'eigen_projection_7_abs_50th',
#  'path_coverage_tail',
#  'width_tail_base_w_backward_10th',
#  'd_eigen_projection_6_IQR',
#  'blob_compactness_50th',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_paused_fraction'],
# ['d_relative_radial_velocity_head_tip_w_forward_90th',
#  'width_head_base_w_backward_50th',
#  'width_head_base_norm_50th',
#  'd_curvature_midbody_w_backward_90th',
#  'd_curvature_head_w_forward_IQR',
#  'curvature_head_w_backward_IQR',
#  'd_eigen_projection_6_IQR',
#  'motion_mode_paused_fraction'],
# ['d_curvature_neck_w_forward_IQR',
#  'width_head_base_w_forward_10th',
#  'width_midbody_norm_10th',
#  'blob_solidity_50th',
#  'd_relative_speed_midbody_10th',
#  'd_eigen_projection_6_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_paused_fraction'],
# ['d_curvature_head_w_forward_IQR',
#  'motion_mode_backward_fraction',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_backward_IQR',
#  'd_eigen_projection_6_IQR',
#  'width_midbody_w_forward_10th',
#  'width_head_base_norm_10th',
#  'quirkiness_w_forward_50th'],
# ['d_blob_solidity_w_forward_90th',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_backward_IQR',
#  'd_curvature_hips_w_backward_10th',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_backward_fraction',
#  'curvature_head_w_forward_IQR',
#  'curvature_tail_norm_abs_90th'],
# ['d_curvature_head_w_forward_IQR',
#  'curvature_head_w_forward_IQR',
#  'width_head_base_norm_10th',
#  'curvature_head_norm_abs_90th',
#  'd_eigen_projection_6_w_forward_IQR',
#  'motion_mode_paused_fraction',
#  'motion_mode_paused_frequency',
#  'width_midbody_w_forward_50th'],
# ['d_width_head_base_IQR',
#  'motion_mode_paused_frequency',
#  'width_midbody_w_forward_10th',
#  'curvature_tail_norm_abs_50th',
#  'd_relative_speed_midbody_90th',
#  'd_eigen_projection_6_w_forward_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_backward_frequency'],
# ['d_blob_hu3_IQR',
#  'length_90th',
#  'motion_mode_backward_fraction',
#  'path_coverage_tail',
#  'd_quirkiness_IQR',
#  'width_head_base_w_forward_10th',
#  'd_curvature_head_w_forward_IQR',
#  'd_eigen_projection_6_w_forward_IQR'],
# ['d_curvature_hips_w_backward_10th',
#  'width_head_base_w_forward_10th',
#  'quirkiness_w_forward_50th',
#  'curvature_head_norm_abs_90th',
#  'd_curvature_head_w_forward_IQR',
#  'blob_solidity_w_forward_50th',
#  'eigen_projection_7_IQR',
#  'motion_mode_paused_fraction'],
# ['d_eigen_projection_6_w_forward_10th',
#  'eigen_projection_2_abs_50th',
#  'width_head_base_w_forward_10th',
#  'd_curvature_midbody_w_backward_90th',
#  'd_curvature_head_w_forward_IQR',
#  'curvature_head_w_forward_IQR',
#  'blob_solidity_50th',
#  'curvature_head_norm_IQR'],
# ['d_curvature_neck_w_forward_IQR',
#  'motion_mode_paused_frequency',
#  'd_relative_speed_midbody_10th',
#  'd_eigen_projection_6_IQR',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_backward_fraction',
#  'd_curvature_midbody_w_backward_90th',
#  'curvature_head_w_forward_IQR'],
# ['d_eigen_projection_7_IQR',
#  'eigen_projection_2_abs_50th',
#  'width_head_base_w_forward_10th',
#  'curvature_head_w_forward_IQR',
#  'd_curvature_midbody_w_backward_90th',
#  'curvature_head_norm_abs_90th',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_paused_fraction'],
# ['d_eigen_projection_6_w_forward_IQR',
#  'curvature_head_abs_90th',
#  'width_head_base_w_forward_10th',
#  'blob_compactness_50th',
#  'd_relative_speed_midbody_10th',
#  'd_eigen_projection_4_10th',
#  'd_curvature_head_w_forward_IQR',
#  'motion_mode_paused_fraction']]
    #%%
    
    df = feat_data['tierpsy']
    k_cols = [x for x in df if x in col2ignore_r]
    for ii, c_set in enumerate(core_sets):
        k = 'core_{}'.format(ii+1)
        feat_data[k] = df[k_cols + list(c_set)]
        
    for ii, c_set in enumerate(rfe_sets):
        k = 'rfe_{}'.format(ii+1)
        feat_data[k] = df[k_cols + list(c_set)]
    #%%
    del feat_data['OW']
    del feat_data['all']
    del feat_data['tierpsy']
    #%%
#    
#    feats2remove = ['angular_velocity',
#                     'motion_mode_paused',
#                     'motion_mode_backward',
#                     'motion_mode_forward',
#                     'path_coverage_tail',
#                     'relative_radial_velocity_hips',
#                     'blob_area',
#                     'path_density_tail',
#                     'turn_intra',
#                     'path_coverage_body',
#                     'path_coverage_midbody',
#                     'turn_inter',
#                     'path_curvature_tail',
#                     'path_transit_time_body',
#                     'path_curvature_head',
#                     'path_coverage_head',
#                     'path_density_midbody',
#                     'path_density_head',
#                     'path_density_body',
#                     'path_transit_time_head',
#                     'path_transit_time_tail',
#                     'path_transit_time_midbody',
#                     'path_curvature_body',
#                     'path_curvature_midbody',
#                     'blob_box_width',
#                     'blob_perimeter'
#                     ]
#    
#    feats2remove2 = ['path_coverage_tail',
#                     'blob_area',
#                     'path_density_tail',
#                     'turn_intra',
#                     'path_coverage_body',
#                     'path_coverage_midbody',
#                     'turn_inter',
#                     'path_curvature_tail',
#                     'path_transit_time_body',
#                     'path_curvature_head',
#                     'path_coverage_head',
#                     'path_density_midbody',
#                     'path_density_head',
#                     'path_density_body',
#                     'path_transit_time_head',
#                     'path_transit_time_tail',
#                     'path_transit_time_midbody',
#                     'path_curvature_body',
#                     'path_curvature_midbody',
#                     'blob_box_width',
#                     'blob_perimeter'
#                     ]
#    #I do not want to use the hu ...
#    df = feat_data['tierpsy']
#    
#    v_cols = [x for x in df.columns if not 'hu' in x]
#    v_cols = [x for x in v_cols if not any(f in x for f in feats2remove)]
#    feat_data['tierpsy_reduced'] = df[v_cols]
#    core_feats['tierpsy_reduced'] = [ x for x in core_feats['tierpsy'] if x not in feats2remove]
#    
#    
#    v_cols = [x for x in df.columns if not 'hu' in x]
#    v_cols = [x for x in v_cols if not any(f in x for f in feats2remove2)]
#    feat_data['tierpsy_reduced2'] = df[v_cols]
#    core_feats['tierpsy_reduced2'] = [ x for x in core_feats['tierpsy'] if x not in feats2remove2]
    
    #%%
#    df = df[v_cols]
#    
#    feat_data['tierpsy_no_hu'] = df
#    core_feats['tierpsy_no_hu'] = core_feats['tierpsy']
#    
#    df = feat_data['tierpsy']
#    dd = [x for x in v_cols if not 'abs' in x]
#    feat_data['cols_no_abs'] = df[dd]
#    core_feats['cols_no_abs'] = core_feats['tierpsy']
#    
#    df = feat_data['tierpsy']
#    dd = [x for x in v_cols if not 'norm' in x]
#    feat_data['cols_no_norm'] = df[dd]
#    core_feats['cols_no_norm'] = core_feats['tierpsy']
#    
#    df = feat_data['tierpsy']
#    dd = [x for x in v_cols if not 'norm' in x and not 'abs' in x]
#    feat_data['cols_no_norm_no_abs'] = df[dd]
#    core_feats['cols_no_norm_no_abs'] = core_feats['tierpsy']
#    
#    
#    df = feat_data['tierpsy']
#    dd = [x for x in v_cols if not '_w_' in x]
#    feat_data['cols_no_subdiv'] = df[dd]
#    core_feats['cols_no_subdiv'] = core_feats['tierpsy']
#    
#    df = feat_data['tierpsy']
#    dd = [x for x in v_cols if not x.startswith('d_')]
#    feat_data['cols_no_dev'] = df[dd]
#    core_feats['cols_no_dev'] = core_feats['tierpsy']
    
    #%%
#    df = feat_data['tierpsy']
#    
#    cols_no_hu = [x for x in df.columns if 'hu' not in x]
#    cols_no_hu_eigen = [x for x in cols_no_hu if 'eigen_projection' not in x]
#    
#    feat_data['tierpsy_no_hu'] = df[cols_no_hu]
#    core_feats['tierpsy_no_hu'] = core_feats['tierpsy']
#    
#    feat_data['tierpsy_no_hu_eigen'] = df[cols_no_hu_eigen]
#    core_feats['tierpsy_no_hu_eigen'] = core_feats['tierpsy']
#    
#    
#    df = feat_data['all']
#    cols_no_hu = [x for x in df.columns if 'hu' not in x]
#    cols_no_hu_eigen = [x for x in cols_no_hu if 'eigen_projection' not in x]
#    feat_data['all_no_hu'] = df[cols_no_hu]
#    core_feats['all_no_hu'] = core_feats['all']
    #%%
    
    
    #%%
    n_folds = 5
    batch_size = 250
    n_epochs = 250
    cuda_id = 0
    
    fold_param = (cuda_id, n_epochs, batch_size)
    
    all_data_in = []
    for db_name, feats in feat_data.items():
        #if db_name != 'OW': continue
        
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        cross_v_res = []
        sss = StratifiedShuffleSplit(n_splits = n_folds, test_size = 0.2, random_state=777)
        for i_fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            x_train, y_train  = X[train_index], y[train_index]
            x_test, y_test  = X[test_index], y[test_index]
            
            fold_data = (x_train, y_train), (x_test, y_test)
            fold_id = (db_name, i_fold)
            
            all_data_in.append((fold_id, fold_data, fold_param))
    #%%
    p = mp.Pool(10)
    results = p.map(simple_fit, all_data_in)
    #%%
    import pandas as pd
    import seaborn as sns
    import matplotlib.pylab as plt
    #%%
    
    #flatten results
    dd = [sum(map(list, x), []) for x in results]    
    
    res_df = pd.DataFrame(dd, columns = ['db', 'fold_n', 'loss', 'acc', 'f1'])
    res_df = res_df[res_df['db']!='tierpsy']
    
    
    fig, ax = plt.subplots(figsize=(12,5))
    #g = sns.swarmplot('db', 'acc', data=res_df)
    
    y_str = 'acc'
    g = sns.swarmplot( 'db', y_str, data=res_df, ax=ax)
    plt.tight_layout()
    #g.get_figure().savefig("global_effects.pdf")
    g.get_figure().savefig("r1_N8_feats_{}.pdf".format(y_str))
    
    
    dd = res_df.groupby('db').agg('mean')
    print(res_df.groupby('db').agg('mean'))