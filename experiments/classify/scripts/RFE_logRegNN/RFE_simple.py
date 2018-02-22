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

from trainer import softmax_RFE
from reader import read_feats

if __name__ == "__main__":
    feat_data, col2ignore_r = read_feats()
    #%%
    n_folds = 10
    batch_size = 250
    
    n_epochs = 250
    metric2exclude = 'loss'
    
    #cuda_id = 0
    #n_feats2remove = 'log2' #1#
    
    cuda_id = 1
    n_feats2remove = 1
    #%%
    fold_param = (cuda_id, n_epochs, batch_size, metric2exclude, n_feats2remove)
    
    all_data_in = []
    for db_name, feats in feat_data.items():
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        
        
        cross_v_res = []
        sss = StratifiedShuffleSplit(n_splits = n_folds, test_size = 0.2, random_state=777)
        for i_fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            x_train, y_train  = X[train_index], y[train_index]
            x_test, y_test  = X[test_index], y[test_index]
            
            fold_data = (x_train, y_train), (x_test, y_test), col_feats.copy()
            fold_id = (db_name, i_fold)
            
            all_data_in.append((fold_id, fold_data, fold_param))
    all_data_in = all_data_in[::-1]
     
    #%%
    p = mp.Pool(10)
    results = p.map(softmax_RFE, all_data_in)
    
    save_name = 'RFE_SoftMax_F{}.pkl'.format(n_feats2remove)
    with open(save_name, "wb" ) as fid:
        pickle.dump(results, fid)
    #%%
    
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    
    
    #res = softmax_RFE(all_data_in[4])
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
        tot = len(sum(feats, []))
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        
        
        if n_feats2remove == 'log2':
            n2 = int(np.floor(np.log2(tot - 1e-5)))
            xx = np.array([tot] + [2**x for x in range(n2, 0, -1)])
        else:
            xx = np.arange(tot, 0, -n_feats2remove) + 1
        
        
        h = ax.errorbar(xx, yy, yerr=err, label = k)
    #plt.xlim(0, 32)
    plt.legend()
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    
    fig.savefig('RFE_comparison.pdf')
    
    #%%
    
    for k, dat in res_db.items():
        #if k != 'OW': continue
        
        plt.figure()
        
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        tot = len(sum(feats, []))
        
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
        feats = [sum(x, []) for x in feats]
        
        
        
        col_feats = [x for x in feat_data[k].columns if x not in col2ignore_r]
        for ff in feats:
            dd = list(set(col_feats) - set(ff))
            assert len(dd) == 1
            ff.append(dd[0])
        
        
        #min_ind = 20
        rr = None
        for ff in feats:
            s = ff[:min_ind]
            if rr is None:
                rr = set(s)
            else:
                rr.intersection_update(s)
        
        rr2 = None
        for ff in feats:
            s = ff[min_ind:]
            if rr2 is None:
                rr2 = set(s)
            else:
                rr2.intersection_update(s)
               
        print(k, tot, min_ind,len(rr), tot-min_ind, len(rr2))
        
        for ff in feats:
            print(ff[-10:])
        print('%%%%%%%%%%%%%%%%%%%%%%%%')
        
    #plt.xlim((0,20))