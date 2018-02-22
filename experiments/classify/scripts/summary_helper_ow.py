#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:04:47 2018

@author: ajaver
"""

import numpy as np
import pandas as pd
import tables

from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormStats
#%%
def read_feat_events(fname):
    
    def _feat_correct(x):
        x = x.replace('_motion', '')
        dat2rep =  [('_upsilon', '_upsilon_turns'),
                    ('upsilon_turn', 'upsilon_turns'),
                    ('_omega', '_omega_turns'),
                    ('omega_turn', '_omega_turns'),
                    ('coil_', 'coils_'),
                    ]
        for p_old, p_new in dat2rep:
            if not (p_new in x) and (p_old in x):
                x = x.replace(p_old, p_new)
        return x
        
    
    with tables.File(fname, 'r') as fid:
        features_events = {}
        node = fid.get_node('/features_events')
        for worn_n in node._v_children.keys():
            worm_node = fid.get_node('/features_events/' + worn_n)
            
            for feat in worm_node._v_children.keys():
                
                dat = fid.get_node(worm_node._v_pathname, feat)[:]
                
                
                feat_c = _feat_correct(feat)
                
                if not feat_c in features_events:
                    features_events[feat_c] = []
                features_events[feat_c].append(dat)
    
    
    features_events = {feat:np.concatenate(val) for feat, val in features_events.items()}
    
    return features_events
#%%
def process_ow_file(fname):
    
    with pd.HDFStore(fname, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
    all_feats = read_feat_events(fname)
    
    for cc in features_timeseries:
        all_feats[cc] = features_timeseries[cc].values
    
    wStats = WormStats()
    exp_feats = wStats.getWormStats(all_feats, np.nanmean)
    
    
    exp_feats = pd.DataFrame(exp_feats).T[0]
    
    valid_c = [x for x in exp_feats.index if x not in wStats.extra_fields]
    exp_feats = exp_feats[valid_c]
    
    return exp_feats
    
#%%
if __name__ == '__main__':
    #fname =  '/Volumes/behavgenom_archive$/single_worm/finished/WT/AQ2947/food_OP50/XX/30m_wait/anticlockwise/483 AQ2947 on food R_2012_03_08__15_42_48___1___8_features.hdf5'
    #fname = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/del-1(ok150)X@NC279/food_OP50/XX/30m_wait/clockwise/del-1 (ok150)X on food L_2012_03_08__15_16_22___1___7_features.hdf5'
    fname = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_020617/N2_worms5_food1-10_Set1_Pos4_Ch5_02062017_115615_features.hdf5'
    exp_feats = process_ow_file(fname)
    
    #make sure all the features are unique
    #assert np.unique(exp_feats.index).size == exp_feats.size
    #%%
    #for x in exp_feats.index:
    #    print(x)
    
    