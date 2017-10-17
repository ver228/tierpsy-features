#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:52:11 2017

@author: ajaver
"""

import glob
import os
import tables
from tierpsy.helper.params import read_ventral_side

main_dir = '/Volumes/behavgenom_archive$/single_worm/finished'

fnames = glob.glob(os.path.join(main_dir, '**', '*_featuresN.hdf5'), recursive=True)

for ii, fname in enumerate(fnames):
    print(ii+1, len(fnames))
    skeletons_file = fname.replace('_featuresN', '_skeletons')
    try:
        with tables.File(fname, 'r+') as fid:
            fid.get_node('/trajectories_data')._v_attrs['ventral_side'] = read_ventral_side(skeletons_file)
    except tables.exceptions.NoSuchNodeError:
        pass