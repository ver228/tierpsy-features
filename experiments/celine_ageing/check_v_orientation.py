#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 18:30:55 2017

@author: ajaver
"""
import os
import pandas as pd
from tierpsy.helper.params import read_fps, read_ventral_side



if __name__ == '__main__':
    experiments_df = pd.read_csv('ageing_celine.csv')
    
    bad_f = []
    good_f = []
    for irow, row in experiments_df.iterrows():
        print(irow, len(experiments_df))
        skel_file  = os.path.join(row['directory'], row['base_name'] + '_skeletons.hdf5')
        print(skel_file)

        ventral_side = read_ventral_side(skel_file)
        if '_CW_' in skel_file and ventral_side != 'clockwise':
            bad_f.append(skel_file)
            print(skel_file)
        elif '_CCW_' in skel_file and ventral_side != 'anticlockwise':
            bad_f.append(skel_file)
            print(skel_file)
        else:
            good_f.append((ventral_side, skel_file))
    #%%
#    import tables
#    
#    for ii, skel_file in enumerate(bad_f):
#        print(ii, len(bad_f))        
#        
#        bn = skel_file.replace('_skeletons.hdf5', '')
#        for f_ext in ['_intensities.hdf5', '.wcon.zip', '_featuresN.hdf5', '_features.hdf5']:
#            ff = bn + f_ext
#            if os.path.exists(ff):
#                fname = os.remove(ff)
#        
#       
#        if '_CW_' in skel_file:
#            r_ventral_side = 'clockwise'
#        elif '_CCW_' in skel_file:
#            r_ventral_side = 'anticlockwise' 
#        else:
#            ValueError()
#                
#        with tables.File(skel_file, 'r+') as fid:
#            fid.get_node('/trajectories_data')._v_attrs['ventral_side'] = r_ventral_side
#            
#        