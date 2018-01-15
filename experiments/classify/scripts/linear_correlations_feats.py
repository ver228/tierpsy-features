#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:54:05 2017

@author: ajaver
"""

import pandas as pd
import numpy as np

if __name__ == '__main__':
    tierpsy_feats = pd.read_csv('tierpsy_features_CeNDR.csv', index_col=False)
    info_cols = ['Unnamed: 0', 'id', 'strain', 'directory', 'base_name', 'exp_name']
    feat_cols = [x for x in tierpsy_feats if x not in info_cols]
    corr_mat = tierpsy_feats[feat_cols].corr()
    #corr_mat = tierpsy_feats[feat_cols].corr('spearman')
    
    gg = [corr_mat.index[corr_mat[col].abs()>0.9] for col in corr_mat]
    gg = [list(x) for x in gg if len(x) > 1]
    
    gg_reduced = []
    for x in gg:
        if not x in gg_reduced:
            gg_reduced.append(x)
    
    gg_reduced = sorted(gg_reduced, key = lambda x : np.abs(corr_mat.loc[x[0], x[1]]))[::-1]
    for x in gg_reduced:
        print(corr_mat.loc[x[0], x[1]], x)
 #%%
 import matplotlib.pylab as plt
 plt.plot(tierpsy_feats['path_density_body_95th'], tierpsy_feats['path_density_head_95th'], '.')