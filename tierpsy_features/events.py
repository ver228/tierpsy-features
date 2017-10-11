#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:41:10 2017

@author: ajaver
"""

import numpy as np
import pandas as pd

#%%
def _get_pulses_indexes(light_on, min_window_size=0, is_pad = True):
    '''
    Get the start and end of a given pulse.
    '''
    
    if is_pad:
        
        light_on = np.pad(light_on, (1,1), 'constant', constant_values = False)
       
    switches = np.diff(light_on.astype(np.int))
    turn_on, = np.where(switches==1)
    turn_off, = np.where(switches==-1)
    
    if is_pad:
        turn_on -= 1
        turn_off -= 1
        turn_on = np.clip(turn_on, 0, light_on.size-3)
        turn_off = np.clip(turn_off, 0, light_on.size-3)
        
    
    assert turn_on.size == turn_off.size
    
    delP = turn_off - turn_on
    
    good = delP > min_window_size
    
    return turn_on[good], turn_off[good]
#%%
def _range_vec(vec, th):
    '''
    flag a vector depending on the threshold, th
    -1 if the value is below -th
    1 if the value is above th
    0 if it is between -th and th
    '''
    flags = np.zeros(vec.size)
    _out = vec < -th
    _in = vec > th
    flags[_out] = -1
    flags[_in] = 1
    return flags

def _flag_regions(vec, central_th, extrema_th, smooth_window, min_frame_range):
    '''
    Flag a frames into lower (-1), central (0) and higher (1) regions.
    
    The strategy is
        1) Smooth the timeseries by smoothed window
        2) Find frames that are certainly lower or higher using extrema_th
        3) Find regions that are between (-central_th, central_th) and 
            and last more than min_zero_window. This regions are certainly
            central regions.  
        4) If a region was not identified as central, but contains 
            frames labeled with a given extrema, label the whole region 
            with the corresponding extrema.
    '''
    vv = pd.Series(vec).fillna(method='ffill').fillna(method='bfill')
    smoothed_vec = vv.rolling(window=smooth_window,center=True).mean()
    
    
    
    paused_f = (smoothed_vec > -central_th) & (smoothed_vec < central_th)
    turn_on, turn_off = _get_pulses_indexes(paused_f, min_frame_range)
    inter_pulses = zip([0] + list(turn_off), list(turn_on) + [paused_f.size-1])
    
    
    flag_modes = _range_vec(smoothed_vec, extrema_th)
    
    for ini, fin in inter_pulses:
        dd = np.unique(flag_modes[ini:fin+1])
        dd = [x for x in dd if x != 0]
        if len(dd) == 1:
            flag_modes[ini:fin+1] = dd[0]
        elif len(dd) > 1:
            kk = flag_modes[ini:fin+1]
            kk[kk==0] = np.nan
            kk = pd.Series(kk).fillna(method='ffill').fillna(method='bfill')
            flag_modes[ini:fin+1] = kk
    return flag_modes

def _get_vec_durations(event_vec):
    durations_list = []
    for e_id in np.unique(event_vec):
        ini_e, fin_e = _get_pulses_indexes(event_vec == e_id, is_pad = True)
        event_durations = fin_e - ini_e
        
        #flag if the event is on the vector edge or not
        edge_flag = np.zeros_like(fin_e)
        edge_flag[ini_e <= 0] = -1
        edge_flag[fin_e >= event_vec.size-1] = 1
        
        event_ids = np.full(event_durations.shape, e_id)
        durations_list.append(np.stack((event_ids, event_durations, ini_e, fin_e, edge_flag)).T)
    
    cols = ['region', 'duration', 'timestamp_initial', 'timestamp_final', 'edge_flag']
    event_durations_df = pd.DataFrame(np.concatenate(durations_list), columns = cols)
    return event_durations_df

def get_event_durations(events_df, fps):
    event_durations_list = []
    for col in events_df:
        if not col in ['timestamp', 'worm_index']:
            dd = _get_vec_durations(events_df[col].values)
            dd.insert(0, 'event_type', col)
            event_durations_list.append(dd)

    if len(event_durations_list) == 0:
        event_durations_df = pd.DataFrame()
    else:

        event_durations_df = pd.concat(event_durations_list, ignore_index=True)
        event_durations_df['duration'] /= fps
        #shift timestamps to match the real initial time
        first_t = events_df['timestamp'].min()
        event_durations_df['timestamp_initial'] += first_t
        event_durations_df['timestamp_final'] += first_t
    
    
    return event_durations_df


def get_events(df, fps, worm_length = None, _is_debug=False):
    
    #initialize data
    smooth_window_s = 0.5
    min_paused_win_speed_s = 1/3
    
    if worm_length is None:
        assert 'length' in df
        worm_length = df['length'].median()
    
    
    df = df.sort_values(by='timestamp')
    
    w_size = int(round(fps*smooth_window_s))
    smooth_window = w_size if w_size % 2 == 1 else w_size + 1
    
    #WORM MOTION EVENTS
    events_df = pd.DataFrame(df[['worm_index', 'timestamp']])
    if 'speed' in df:
        speed = df['speed'].values
        pause_th_lower = worm_length*0.025
        pause_th_higher = worm_length*0.05
        min_paused_win_speed = fps/min_paused_win_speed_s

        motion_mode = _flag_regions(speed, 
                                 pause_th_lower, 
                                 pause_th_higher, 
                                 smooth_window, 
                                 min_paused_win_speed
                                 ) 
        events_df['motion_mode'] = motion_mode
    
    #FOOD EDGE EVENTS
    if 'dist_from_food_edge' in df:
        dist_from_food_edge = df['dist_from_food_edge'].values
        edge_offset_lower = worm_length/2
        edge_offset_higher = worm_length
        min_paused_win_food_s = 1
        
        min_paused_win_food = fps/min_paused_win_food_s
        food_region = _flag_regions(dist_from_food_edge, 
                                     edge_offset_lower, 
                                     edge_offset_higher, 
                                     smooth_window, 
                                     min_paused_win_food
                                     ) 
        events_df['food_region'] = food_region
    
    if _is_debug:
        plt.figure()
        plt.plot(speed)
        plt.plot(motion_mode*pause_th_higher)
        
        plt.figure()
        plt.plot(dist_from_food_edge)
        plt.plot(food_region*edge_offset_lower)
        
    
    #get event durations
    event_durations_df = get_event_durations(events_df, fps)
    
    return events_df, event_durations_df       



if __name__ == '__main__':
    from tierpsy.helper.params import read_fps
    import matplotlib.pylab as plt
    import os
    import glob

    
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    fnames = glob.glob(os.path.join(dname, 'Experiment8', '**', '*_featuresN.hdf5'), recursive = True)
    
    for ifname, fname in enumerate(fnames):
        print(ifname+1, len(fnames))
        with pd.HDFStore(fname, 'r') as fid:
            if '/provenance_tracking/FEAT_TIERPSY' in fid:
                timeseries_features = fid['/timeseries_features']
            else:
                continue
        break
    
    #%%
    fps = read_fps(fname)
    for w_index in [2]:#, 69, 431, 437, 608]:
        worm_data = timeseries_features[timeseries_features['worm_index']==w_index]
        worm_length = worm_data['length'].median()
        
        events_df, event_durations_df = get_events(worm_data, fps, _is_debug=True)
    
        #%%
        #angular_velocity = timeseries_features.loc[good, 'angular_velocity'].values
        angular_velocity = timeseries_features['angular_velocity'].values
        smooth_window = int(round(fps*2))  
        
        dd = angular_velocity.copy()
        dd[np.isnan(dd)] = 0
        dd = dd.cumsum()
        
        #smoothed_vec = pd.Series(dd).rolling(window=smooth_window,center=True).mean()
        
        #%%
        plt.figure()
        worm_data['curvature_tail'].plot()
        worm_data['curvature_midbody'].plot()
        worm_data['curvature_head'].plot()
    #%%
    
    
    
    
    
    