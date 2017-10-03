#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:55:02 2017

@author: ajaver
"""
import pandas as pd
import glob
import os
from tierpsy_features.summary_stats import collect_feat_stats


if __name__ == '__main__':
    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    
    exp_groups = [
        dict(
            set_prefix = 'Vortex',
            exp_valid = [5,6],
            time_ranges_m = [(0, 7.5), (7.5, 14), (16, 22.5), (22.5, 30)],
            is_normalize_data = False
        ),
        dict(
            set_prefix = '4h',
            exp_valid = [7,8],
            time_ranges_m = [(n*15, (n+1)*15) for n in range(8)],
            is_normalize_data = False
        )
    ]
    
    all_fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive = True)
    
    
    is_normalize = True
    for exp_g in exp_groups:
        
        fnames = [x for x in all_fnames if int(x.split('Experiment')[-1].split('/')[0]) in exp_g['exp_valid']]
        
        set_prefix = exp_g['set_prefix']
        if is_normalize:
            set_prefix += '_N'
        
        dat = []
        for fname in fnames:
            exp_key = fname.split(os.sep)[-2].split('_')[-1]
            exp_n = int(exp_key.split('co')[0][3:])
            cohort_n = int(exp_key.split('co')[-1])
            dat.append((exp_n, cohort_n))
        info_df = pd.DataFrame(dat, columns = ['exp_n', 'cohort_n'])
        
        
        assert info_df.shape[0] == len(fnames)
        
        feat_means_df = collect_feat_stats(fnames, 
                                          info_df, 
                                          time_ranges_m = exp_g['time_ranges_m'],
                                          is_normalize = is_normalize
                                          )
        feat_means_df.to_csv(set_prefix + '_data.csv', index=False)
        