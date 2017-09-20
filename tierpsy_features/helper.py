#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:36:52 2017

@author: ajaver
"""

import numpy as np
import numba


@numba.jit
def fillfnan(arr):
    '''
    fill foward nan values (iterate using the last valid nan)
    I define this function so I do not have to call pandas DataFrame
    '''
    out = arr.copy()
    for idx in range(1, out.shape[0]):
        if np.isnan(out[idx]):
            out[idx] = out[idx - 1]
    return out

@numba.jit
def fillbnan(arr):
    '''
    fill foward nan values (iterate using the last valid nan)
    I define this function so I do not have to call pandas DataFrame
    '''
    out = arr.copy()
    for idx in range(out.shape[0]-1)[::-1]:
        if np.isnan(out[idx]):
            out[idx] = out[idx+1]
    return out

@numba.jit
def nanunwrap(x):
    '''correct for phase change for a vector with nan values 
    '''
    bad = np.isnan(x)
    x = fillfnan(x)
    x = np.unwrap(x)
    x[bad] = np.nan
    return x

class DataPartition():
    def __init__(self, partitions=None, n_segments=49):
        partitions_dflt = {'head': (0, 8),
                            'neck': (8, 16),
                            'midbody': (16, 33),
                            'hips': (33, 41),
                            'tail': (41, 49),
                            'head_tip': (0, 3),
                            'head_base': (5, 8),
                            'tail_base': (41, 44),
                            'tail_tip': (46, 49),
                            'all': (0, 49),
                            'hh' : (0, 16),
                            'tt' : (33, 49),
                            'body': (8, 41),
                            }
        
        if partitions is None:
            partitions = partitions_dflt
        else:
            partitions = {p:partitions_dflt[p] for p in partitions}
            
        
        if n_segments != 49:
            r_fun = lambda x : int(round(x/49*n_segments))
            for key in partitions:
                partitions[key] = tuple(map(r_fun, partitions[key]))
        
        self.n_segments = n_segments
        self.partitions =  partitions

    def apply(self, data, partition, func, segment_axis=1):
        assert self.n_segments == data.shape[segment_axis]
        assert partition in self.partitions
        
        ini, fin = self.partitions[partition]
        sub_data = np.take(data, np.arange(ini, fin), axis=segment_axis)
        d_transform = func(sub_data, axis=segment_axis)
        
        return d_transform
   
    def apply_partitions(self, data, func, segment_axis=1):
        return {p:self.apply(data, p, func, segment_axis=segment_axis) for p in self.partitions}
