#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:49:57 2018

@author: ajaver
"""
import pandas as pd
import pytz
from collections import OrderedDict

experimenter_dflt = 'Céline N. Martineau, Bora Baskaner'
lab_dflt = {'name' : 'European Research Institute for the Biology of Ageing',  
            'location':'The Netherlands'}
timezone_dflt = 'Europe/Amsterdam'
media_dflt = "NGM agar low peptone"
food_dflt = "OP50"
sex_dflt = "hermaphrodite"

def db_row2dict(row):
    experiment_info = OrderedDict()
    experiment_info['base_name'] = row['base_name']
    experiment_info['who'] = experimenter_dflt
    experiment_info['lab'] = lab_dflt
    
    #add timestamp with timezone
    local = pytz.timezone(timezone_dflt) 
    local_dt = local.localize(row['timestamp'], is_dst=True)
    experiment_info['timestamp'] = local_dt.isoformat()
    
    experiment_info['arena'] = {
            "style":'petri',
            "size":35,
            "orientation":"toward"
            }
    
    experiment_info['media'] = media_dflt
    experiment_info['food'] = food_dflt
    
    experiment_info['strain'] = row['strain']
    experiment_info['strain_description'] = row['strain_description']
    
    
    experiment_info['sex'] = sex_dflt
    experiment_info['stage'] = row['developmental_stage']
    if experiment_info['stage'] == "young adult":
        experiment_info['stage'] = 'adult'
    
    experiment_info['ventral_side'] = row['ventral_side']
    
    
    if row['habituation'] == 'NONE':
        hab = "no wait before recording starts."
    else:
        hab = "worm transferred to arena 30 minutes before recording starts."
    experiment_info['protocol'] = [
        "method in E. Yemini et al. doi:10.1038/nmeth.2560",
        hab
    ]
    
    experiment_info['habituation'] = row['habituation']
    experiment_info['tracker'] = row['tracker']
    experiment_info['original_video_name'] = row['original_video']
    
    return experiment_info



strains_data = [
    ('OW939', 'zgIs113[P(dat-1)::alpha-Synuclein::YFP]', 1, 1, 1),
    ('OW940', 'zgIs128[P(dat-1)::alpha-Synuclein::YFP]', 1, 1, 1),
    ('OW949', 'zgIs125[P(dat-1)::alpha-Synuclein::YFP]', 1, 1, 1),
    ('OW953', 'zgIs138[P(dat-1)::YFP]', 1, 1, 1),
    ('OW956', 'zgIs144[P(dat-1)::YFP]', 1, 1, 1)
    ]
    
if __name__ == '__main__':
    exp_file = 'ageing_celine.csv'
    experiments_df = pd.read_csv(exp_file)
    
    #use the strain code from the CGC
    experiments_df['strain'] = experiments_df['strain'].replace({'N2' : 'AQ2947'})
    experiments_df = experiments_df.rename(columns={"ventral_orientation": "ventral_side"})
    
    
    experiments_df['ventral_side'] = experiments_df['ventral_side'].replace({'CW':'clockwise', 'CCW':'anticlockwise'})
    
    u_strain = experiments_df['strain'].unique()
    u_day = experiments_df['day'].unique()
    
    for _, row in experiments_df.iterrows():
        
        row_d = row.to_dict()
        del row_d['id']
        del row_d['replicated_n']
        break
    
    
    
#['directory', 'base_name', 'replicated_n', 'strain', 'worm_id', 'day',
#       'ventral_orientation', 'timestamp', 'id']
#+------------------------+--------------+------+-----+---------+----------------+
#| Field                  | Type         | Null | Key | Default | Extra          |
#+------------------------+--------------+------+-----+---------+----------------+
#| id                     | int(11)      | NO   | PRI | NULL    | auto_increment |
#| base_name              | varchar(200) | NO   | UNI | NULL    |                |
#| date                   | datetime     | YES  |     | NULL    |                |
#| strain_id              | int(11)      | YES  | MUL | NULL    |                |
#| tracker_id             | int(11)      | YES  | MUL | NULL    |                |
#| sex_id                 | int(11)      | YES  | MUL | NULL    |                |
#| developmental_stage_id | int(11)      | YES  | MUL | NULL    |                |
#| ventral_side_id        | int(11)      | YES  | MUL | NULL    |                |
#| food_id                | int(11)      | YES  | MUL | NULL    |                |
#| arena_id               | int(11)      | YES  | MUL | NULL    |                |
#| habituation_id         | int(11)      | YES  | MUL | NULL    |                |
#| experimenter_id        | int(11)      | YES  | MUL | NULL    |                |
#| original_video         | varchar(700) | NO   | UNI | NULL    |                |
#| original_video_sizeMB  | float        | YES  |     | NULL    |                |
#| exit_flag_id           | int(11)      | NO   | MUL | 0       |                |
#| results_dir            | varchar(200) | YES  |     | NULL    |                |
#| youtube_id             | varchar(40)  | YES  |     | NULL    |                |
#+------------------------+--------------+------+-----+---------+----------------+
