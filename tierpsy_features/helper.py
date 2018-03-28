#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:36:52 2017

@author: ajaver
"""
import pandas as pd
import numpy as np
import numba
import math

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
    x = fillbnan(x)
    x = np.unwrap(x)
    x[bad] = np.nan
    return x

def get_n_worms_estimate(frame_numbers, percentile = 99):
    '''
    Get an estimate of the number of worms using the table frame_numbers vector
    '''
    
    n_per_frame = frame_numbers.value_counts()
    n_per_frame = n_per_frame.values
    if len(n_per_frame) > 0:
        n_worms_estimate = np.percentile(n_per_frame, percentile)
    else:
        n_worms_estimate = 0
    return n_worms_estimate


def get_delta_in_frames(delta_time, fps):
    '''Get the conversion of delta time in frames. Make sure it is more than one.'''
    return max(1, int(round(fps*delta_time)))

def add_derivatives(feats, cols2deriv, delta_frames, fps):
    '''
    Calculate the derivatives of timeseries features, and add the columns to the original dataframe.
    '''
    #%%
    val_cols = [x for x in cols2deriv if x in feats]
    
    feats = feats.sort_values(by='timestamp')
    
    df_ts = feats[val_cols].copy()
    df_ts.columns = ['d_' + x for x in df_ts.columns]
    
    m_o, m_f = math.floor(delta_frames/2), math.ceil(delta_frames/2)
    
    
    vf = df_ts.iloc[delta_frames:].values
    vo = df_ts.iloc[:-delta_frames].values
    vv = (vf - vo)/(delta_frames/fps)
    
    #the series was too small to calculate the derivative
    if vv.size > 0:
        df_ts.loc[:] =  np.nan
        df_ts.iloc[m_o:-m_f] = vv
        
    feats = pd.concat([feats, df_ts], axis=1)
    #%%
    return feats

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
