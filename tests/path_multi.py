#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:01:01 2017

@author: ajaver
"""
import pandas as pd


if __name__ == '__main__':
    fname = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/RIG_HDF5_VIDEOS/Results/RIG_HDF5_VIDEOS_featuresN.hdf5'
    
    with pd.HDFStore(fname, 'r') as fid:
        blob_features = fid.get_node('/blob_features')[:]
        trajectories_data = fid.get_node('/trajectories_data')[:]