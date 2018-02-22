#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:58:24 2018

@author: ajaver
"""
import pickle
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
if __name__ == '__main__':
    save_name = 'RFE_G_SoftMax.pkl'
   
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
    feats_div = {}
    for db_name, fold_data in res_db.items():
        #if k != 'tierpsy': continue
        
        plt.figure()
        
        dd = []
        for (feats, loss, acc, f1) in fold_data:
            dd.append(acc)
        tot = len(feats)
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        plt.errorbar(xx, yy, yerr=err)
        
        ind = np.argmax(yy)
        
        th = yy[ind] - err[ind]
        min_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[min_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(db_name, x_t, yy[min_ind], yy.max())
        
        plt.title(db_name)
        
        
        feat_orders = {}
        
        for dat in fold_data:
            for ii, f in enumerate(dat[0]):
                if not f in feat_orders:
                    feat_orders[f] = []
                feat_orders[f].append(ii)
        
        feats, order_vals = zip(*feat_orders.items())
        df = pd.DataFrame(np.array(order_vals), index=feats)
        df_m = df.median(axis=1).sort_values()
        
    
        
        useless_feats = df_m.index[:min_ind]
        usefull_feats = df_m.index[min_ind:]
        feats_div[db_name] = (useless_feats, usefull_feats)
        #useless_feats = sum([list(x[:min_ind]) for x in feats], [])
        #usefull_feats = sum([list(x[min_ind:]) for x in feats], [])
        #useless_feats = sorted(Counter(useless_feats).items(), key = lambda x : x[1])[::-1]
        #usefull_feats = sorted(Counter(usefull_feats).items(), key = lambda x : x[1])[::-1]
    #%%
    useless_feats, usefull_feats = feats_div['tierpsy']
    
    
    ff_str = 'turn'
    dd = [(ii,x) for ii,x in enumerate(useless_feats) if ff_str in x]
    print('BAD *****', dd)
    
    dd = [(ii,x) for ii,x in enumerate(usefull_feats) if ff_str in x]
    print('GOOD ****', dd)