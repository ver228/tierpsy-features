#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:26:15 2018

@author: ajaver
"""

import numpy as np
from reader import read_feats
import pandas as pd
from scipy.stats import ttest_ind, ranksums

core_feats = [
         'length_50th',
         'width_midbody_norm_50th',
         'curvature_hips_abs_90th',
         'curvature_head_abs_90th',
         'motion_mode_backward_fraction',
         'motion_mode_forward_frequency',
         'd_curvature_hips_90th',
         'd_curvature_head_90th',
         ]

if __name__ == "__main__":
    feat_data, col2ignore_r = read_feats() 
    
    df = feat_data['tierpsy']
    #df = df[['date', 'strain']+ maybe_good].copy()
    
    df['date'] = pd.to_datetime(df['date'])
    
    df_g = df.groupby('strain')
    
    N2_data = df_g.get_group('N2')
    
    time_offset_allowed = 7
    
    valid_feats = [x for x in df if x not in col2ignore_r]
    
    ctr_sizes = []
    all_pvalues = []
    for ss, s_data in df_g:
        print(ss)
        if ss == 'N2':
            continue
        
        offset = pd.to_timedelta(time_offset_allowed, unit='day')
        #ini_date = s_data['date'].min() - offset
        #fin_date = s_data['date'].max() + offset
        
        udates = s_data['date'].map(lambda t: t.date()).unique()
        udates = [pd.to_datetime(x) for x in udates]
        
        good = (N2_data['date'] > udates[0] - offset) & (N2_data['date'] < udates[0] + offset)
        for ud in udates:
            good |= (N2_data['date'] > ud - offset) & (N2_data['date'] < ud + offset)
        ctrl_data = N2_data[good]
        
        ctr_sizes.append((ss, len(ctrl_data)))
        
        
        for ff in valid_feats:
            ctr = ctrl_data[ff].values
            atr = s_data[ff].values
            
            #_, p = ttest_ind(ctr, atr)
            _, p = ranksums(ctr, atr)
            assert isinstance(p, float)
            all_pvalues.append((ss, ff, p))
    #%%
    df_pvalues = pd.DataFrame(all_pvalues, columns = ['strain', 'feature', 'pvalue'])
    df_pvalues = df_pvalues.pivot('strain', 'feature', 'pvalue')
    #%%
    pp = df_pvalues*(df_pvalues.shape[0]*df_pvalues.shape[1]) #bonferroni correction
    n_significative = ((pp<0.05).sum(axis = 0)).sort_values()
    n_strains = ((pp<0.05).sum(axis = 1)).sort_values()