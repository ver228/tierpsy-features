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
from reader import read_feats, get_core_features, get_feat_group_indexes




#%%
if __name__ == "__main__":
    feat_data, col2ignore_r = read_feats()
    core_feats = get_core_features(feat_data, col2ignore_r)
    
    df = feat_data['all']
    cols_no_hu = [x for x in df.columns if 'hu' not in x]
    feat_data['all'] = df[cols_no_hu]
    
    
    del feat_data['OW'], feat_data['tierpsy']
    
    #%%
    n_folds = 10
    batch_size = 250
    
    n_epochs = 250
    
    cuda_id = 1
    
    fold_param = (cuda_id, n_epochs, batch_size)
    
    all_data_in = []
    for db_name, feats in feat_data.items():
        #if db_name != 'OW': continue
        
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        feats_groups_inds, core_feats_dict = get_feat_group_indexes(core_feats[db_name], col_feats)
        
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        cross_v_res = []
        sss = StratifiedShuffleSplit(n_splits = n_folds, test_size = 0.2, random_state=777)
        for i_fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            x_train, y_train  = X[train_index], y[train_index]
            x_test, y_test  = X[test_index], y[test_index]
            
            fold_data = (x_train, y_train), (x_test, y_test), (feats_groups_inds, core_feats_dict)
            fold_id = (db_name, i_fold)
            
            all_data_in.append((fold_id, fold_data, fold_param))
        
        
    #softmax_RFE_g(all_data_in[0])
    #%%    
#def softmax_RFE_g(data_in):
    data_in = all_data_in[0]
    str2include = 'ow_'
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), (feats_groups_inds, core_feats_dict) = fold_data
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
    
    core_feats_l = list(core_feats_dict.keys())
    core_feats_l = [x for x in core_feats_l if str2include in x]
    
    fold_res = []
    
    
    while core_feats_l: #continue as long as there are any feature to remove
        n_features = input_v.size(1)
        trainer = TrainerSimpleNet(n_classes, n_features, n_epochs, batch_size, cuda_id)
        trainer.fit(input_train, target_train)
        res = trainer.evaluate(input_v, target_v)
        
        print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*res))
        
        n_core_f = len(core_feats_l)
        print(i_fold + 1, n_core_f, n_features)
        
        
        if n_core_f < 2:
            fold_res.append((core_feats_l[0], res))
            core_feats_l = []
        else:
            # get group importance by removing the features from the model and calculating the result
            
            model = trainer.model
            n_features = model.fc.in_features
            
            res_selection = []
            for f_core in core_feats_l:
                ind = core_feats_dict[f_core]
                ind_valid,  = np.where(feats_groups_inds != ind)
                ind_valid = ind_valid.tolist()
                
                n_features_r = len(ind_valid)
                model_reduced = SimpleNet(n_features_r, n_classes)
                model_reduced.eval()
                
                model_reduced.fc.weight.data = model.fc.weight[:, ind_valid].data
                input_r = input_v[:, ind_valid]
                
                loss, acc, f1 = trainer._evaluate(model_reduced, input_r, target_v)
                
                #i am only using the acc to do the feature selection
                
                res_selection.append((f_core, loss))
            
            #select the group that has the least detrimental effect after being removed 
            group2remove = min(res_selection, key= lambda x : x[1])[0]
            ind = core_feats_dict[group2remove]
            ind_valid,  = np.where(feats_groups_inds != ind)
            ind_valid = ind_valid.tolist()
            
            assert len(ind_valid) > 0
            
            input_v = input_v[:, ind_valid]
            input_train = input_train[:, ind_valid]
            feats_groups_inds = feats_groups_inds[ind_valid]
            core_feats_l.remove(group2remove)
            
            assert input_v.size(1) < n_features
            
            #add progress
            fold_res.append((group2remove, res))
        
    #return fold_id, fold_res

