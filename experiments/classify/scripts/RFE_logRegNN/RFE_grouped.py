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
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import multiprocessing as mp


#%%
import torch
from torch.autograd import Variable
from helper import TrainerSimpleNet, SimpleNet, col2ignore

#%%    
def softmax_RFE_g(data_in):
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), (feats_groups_inds, core_feats_dict) = fold_data
    (cuda_id, n_epochs, batch_size, n_feats2remove) = fold_param
    
    
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
        #%%
    return fold_id, fold_res

#%%
def remove_end(col_v, p2rev):
    col_v_f = []
    for x in col_v:
        xf = x
        for p in p2rev:
            if x.endswith(p):
                xf = xf[:-len(p)]
                break
        col_v_f.append(xf)
    
    return list(set(col_v_f))
#%%
def get_feat_group_indexes(core_feats_v, col_feats):
    '''
    Get the keys, i am assuming there is a core_feats for each of the col_feats
    '''
    
    c_feats_dict = {x:ii for ii, x in enumerate(core_feats_v)}
    
    #sort features by length. In this way I give priority to the longer feat e.g area_length vs area 
    core_feats_v = sorted(core_feats_v, key = len)[::-1]
    
    def _search_feat(feat):
        for core_f in core_feats_v:
            if feat.startswith(core_f):
                return c_feats_dict[core_f]
        print(feat)
        raise('I should not be here... are you sure the keys match?')
    
    col_feats = [x[2:] if x.startswith('d_') else x for x in col_feats]
    f_groups_inds = np.array([_search_feat(x) for x in col_feats])
    
    
    return f_groups_inds, c_feats_dict

#%%
if __name__ == "__main__":
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../../data/SWDB'
    feat_files = {
            'tierpsy' : 'F0.025_tierpsy_features_full_SWDB.csv',
            'OW' : 'F0.025_ow_features_old_SWDB.csv',
            }
    
    #%%
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        #maybe i should divided it in train and test, but cross validation should be enough...
        feats['set_type'] = ''
        feat_data[db_name] = feats
        
    col2ignore_r = col2ignore + ['strain_id', 'set_type']
    
    #%% create a dataset with all the features
    feats = feat_data['OW']
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    feats = feats[col_feats + ['base_name']]
    feats.columns = [x if x == 'base_name' else 'ow_' + x for x in feats.columns]
    feat_data['all'] = feat_data['tierpsy'].merge(feats, on='base_name')
    
    #%% scale data
    for db_name, feats in feat_data.items(): 
        col_val = [x for x in feats.columns if x not in col2ignore_r]
        dd = feats[col_val]
        z = (dd-dd.mean())/(dd.std())
        feats[col_val] = z
        feat_data[db_name] = feats
    
    #%% obtain the core features from the feature list
    core_feats = {}
    
    #OW
    col_v = [x for x in feat_data['OW'].columns if x not in col2ignore_r]
    col_v = remove_end(col_v, ['_abs', '_neg', '_pos'])
    col_v = remove_end(col_v, ['_paused', '_forward', '_backward'])
    col_v = remove_end(col_v, ['_distance', '_distance_ratio', '_frequency', '_time', '_time_ratio'])
    core_feats['OW'] = sorted(col_v)
    
    #tierpsy
    col_v = [x for x in feat_data['tierpsy'].columns if x not in col2ignore_r]
    col_v = list(set([x[2:] if x.startswith('d_') else x for x in col_v]))
    col_v = remove_end(col_v, ['_10th', '_50th', '_90th', '_95th', '_IQR'])
    col_v = remove_end(col_v, ['_w_forward', '_w_backward']) #where is paused??
    col_v = remove_end(col_v, ['_abs'])
    col_v = remove_end(col_v, ['_norm'])
    col_v = remove_end(col_v, ['_frequency', '_fraction', '_duration', ])
    core_feats['tierpsy'] = sorted(col_v)
    
    #all
    core_feats['all']  = core_feats['tierpsy'] + ['ow_' + x for x in core_feats['OW']]
    #%%
    n_folds = 5
    batch_size = 250
    
    n_epochs = 50
    
    cuda_id = 1
    n_feats2remove = 'log2' #1#
    
    fold_param = (cuda_id, n_epochs, batch_size, n_feats2remove)
    
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
    p = mp.Pool(15)
    results = p.map(softmax_RFE_g, all_data_in)
    
    save_name = 'RFE_G_SoftMax.pkl'
    with open(save_name, "wb" ) as fid:
        pickle.dump(results, fid)
    #%%
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    #%%
    res_db = {}
    for (db_name, i_fold), dat in results:
        if db_name not in res_db:
            res_db[db_name] = []
            
        feats, vals = zip(*dat)
        loss, acc, f1 = map(np.array, zip(*vals))
        
        res_db[db_name].append((feats, loss, acc, f1))
    #%%
    import matplotlib.pyplot as plt
    for k, dat in res_db.items():
        plt.figure()
        for (feats, loss, acc, f1) in dat:
            plt.plot(acc)
        plt.title(k)
        
        plt.ylim((0, 55))
    
    plt.figure()
    #%%
    fig, ax = plt.subplots(1, 1)
    for k, dat in res_db.items():
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        
        tot = len(feats)
        xx = np.arange(tot, 0, -1)
        
        h = ax.errorbar(xx, yy, yerr=err, label = k)
    #plt.xlim(0, 32)
    plt.legend()
    #%%
    
    for k, dat in res_db.items():
        #if k != 'OW': continue
        
        plt.figure()
        
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        #tot = len(sum(feats, []))
        tot = len(feats)
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        plt.errorbar(xx, yy, yerr=err)
        
        
        ind = np.argmax(yy)
        #x_max = xx[ind]
        #plt.plot((x_max, x_max), plt.ylim())
        
        
        th = yy[ind] - err[ind]
        min_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[min_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(k, x_t, yy[min_ind])
        
        plt.title(k)
    
        
        feats = [x[0] for x in dat]
        #feats = [sum(x, []) for x in feats]
        #%%
        #col_feats = [x for x in feat_data[k].columns if x not in col2ignore_r]
        #for ff in feats:
        #    dd = list(set(col_feats) - set(ff))
        #    ff.append(dd[0])