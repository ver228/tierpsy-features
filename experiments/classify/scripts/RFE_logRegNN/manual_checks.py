#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:58:24 2018

@author: ajaver
"""
import pickle
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    n_feats2remove = 'log2'
    
    save_name = 'RFE_SoftMax_F{}.pkl'.format(n_feats2remove)
    with open(save_name, "rb" ) as fid:
            results = pickle.load(fid)
    res_db = {}
    for (db_name, i_fold), dat in results:
        if db_name not in res_db:
            res_db[db_name] = []
            
        feats, vals = zip(*dat)
        loss, acc, f1 = map(np.array, zip(*vals))
        
        res_db[db_name].append((feats, loss, acc, f1))
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
    #%%
    #feats = [x[0] for x in res_db['all']]
    
    
    feats = [x[0] for x in res_db['tierpsy']]
    