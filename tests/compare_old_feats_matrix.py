#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:05:32 2017

@author: ajaver
"""
import pandas as pd
import pymysql

if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', database='single_worm_db')
    
    sql = '''
    SELECT e.strain, e.date, feat_m.* 
    FROM experiments_valid AS e
    JOIN features_means AS feat_m ON e.id = feat_m.experiment_id
    WHERE total_time < 905
    AND total_time > 295
    AND n_valid_skeletons > 120*fps
    '''
    
    df = pd.read_sql(sql, con=conn)
    
    #%%
    
    from scipy.stats import kruskal
    
    index_feats = ['strain', 'experiment_id', 'worm_index', 
                   'n_frames', 'n_valid_skel', 'first_frame', 'date']
    feats2check = [x for x in df.columns if not x in index_feats]
    
    results = []
    for feat in feats2check:
        feat_data = df[['strain', feat]]
        dat = [(s, gg[feat].dropna().values) for s, gg in feat_data.groupby('strain')]
        dat = [(s,x) for s,x in dat if len(x) > 10]
        if dat:
            strain_names, samples = zip(*dat)
            
            #k = f_oneway(*samples)
            k = kruskal(*samples)
            
            results.append((feat, k.pvalue, k.statistic))
            
    #%%
    p_th = 0.05/len(feats2check) #0.05 corrected using bonferroni correction
    
    results_df = pd.DataFrame(results, columns = ['feat', 'pvalue', 'statistics'])
    missing_feats = set(feats2check) - set(results_df['feat'])
    bad_feats = results_df[results_df['pvalue']>p_th].sort_values('pvalue')
    
    good_feats = results_df[results_df['pvalue']<p_th]
    
    ## MISSING FEATURES ###
    # **coils** feature is broken
    # **crawling** do not occur when the worm is paused
    # **upsilon** and **omega** cannot be negative
    [x for x in missing_feats if not any(f in x for f in ['crawling_amplitude_paused', 'crawling_frequency_paused', 'coils'])]
    
    # **path_curvature** has a sign that does not depend on the ventral/dorsal orientation, so 
    # only abs/neg/pos give a valid value
    # **orientation** is defined with respect to the field of view, so it is useless (unless there was some sort of chemotaxis)
    # **motion_direction_paused** does not really change when the worm is paused
    [x for x in bad_feats['feat'] if not any(f in x for f in ['orientation', 'path_curvature'])]
    
    #%%
    def _get_valid_ctrs(s_data, min_ctr_size = 10, ini_offset_days = 10):
        exp_dates = (s_data['date'].dt.date).unique()
        
        good = []
        
        
        for day in exp_dates:
            offset_days = ini_offset_days
            while True:
                d_r = (
                    pd.DateOffset(-offset_days)(day),
                    pd.DateOffset(offset_days)(day)
                    )
                g = (ctr_data['date']>=d_r[0]) & (ctr_data['date']<=d_r[1])
                
                if g.sum() < min_ctr_size:
                    offset_days += 1
                else:
                    break
            
            
            
            if len(good) == 0:
                good = g
            else:
                good |= g
        
        return good
    #%%
    from scipy.stats import ranksums
    import statsmodels.stats.multitest as smm
    
    strain_g = df.groupby('strain')
    ctr_data = strain_g.get_group('N2')
    
    r_pvalues = []
    
    n_ctr = []
    for ii, (strain, s_data) in enumerate(strain_g):
        if strain == 'N2':
            continue
        
        print(ii+1, len(strain_g), strain)
        
        good = _get_valid_ctrs(s_data)
        ctr_valid = ctr_data[good]
    
        
        n_ctr.append((strain, ctr_valid.shape[0]))
        
        for feat in good_feats['feat']:
            x = ctr_valid[feat].dropna().values
            y = s_data[feat].dropna().values
            
            if len(y) < 10:
                continue
            
            k = ranksums(x,y)
            r_pvalues.append((strain, feat, k.pvalue))
            
    r_pvalues = pd.DataFrame(r_pvalues, columns=['strain', 'feat', 'pvalue'])
    pvalues_mat = r_pvalues.pivot(index='feat', columns='strain', values='pvalue')
    
    #%%
    
    
    pvalues_mat_corr = pvalues_mat.copy()
    
    for strain in pvalues_mat:
        dat = pvalues_mat[strain]
        good = ~dat.isnull()
        
        pvals = dat[good].values
        reject, pvals_corrected, alphacSidak, alphacBonf = \
            smm.multipletests(pvals, method = 'fdr_tsbky')
        
        pvalues_mat_corr.loc[good, strain] = pvals_corrected
    
    #%%
    
    number_d = pvalues_mat_corr.apply(lambda x : (x<0.05).sum())
    
    #weird
    number_d[number_d==0]
    
    #[VC1759, RB1990, AQ2197, AQ2153]
    #%%
    #TEST USING A WINDOW OF N2
    
    '''
    select s.name, g.name, a.name, description from strains as s join genes AS g on gene_id = g.id join alleles as a on allele_id=a.id  where s.name in ('AQ2153', 'AX1743', 'VC1759', 'VC224');
    '''
    
    
    '''
    select CONCAT(results_dir, '/', base_name, '.hdf5') from experiments_valid where strain in ('AQ2153', 'AX1743', 'VC1759', 'VC224');
    '''
    
    #%%
    mean_old_file = '/Users/ajaver/OneDrive - Imperial College London/single_worm_db/nmeth.2560-S6.xlsx'
    
    mean_old = pd.read_excel(mean_old_file)
    #%%
    bad = ['(+/- = D/V Inside)', '(+/- = Previous D/V)', '(+/- = Toward D/V)', '(+/- = Forward/Backward)', '.']
    
    #tail_bend_mean_backward_pos
    sign_dict = {'positive':'_pos', 
                 'absolute':'_abs', 
                 'negative':'_neg'}
    direction_dict = {'paused':'_paused', 
                      'backward':'_backward', 
                      'forward' : '_forward'}
    
    
    old_names = mean_old.columns
    
    old_rem = []
    for x in old_names:
        for b in bad:
            x = x.replace(b, '')
        old_rem.append(x)
    
    old_rem = [x.lower().split(' ') for x in old_rem]
    old_rem = [[d for d in x if d] for x in old_rem]
    
    
    new_names = []
    for x in old_rem:
        prefix = []
        direction_str = ''
        sign_str = ''
        for p in x:
            if '/' in p:
                p = p.replace('/', '_')+'_ratio'
            
            if ((not any(mm in x for mm in ['inter', 'distance', 'time', 'ratio', 'motion'])) \
            or (any(mm in x for mm in ['amplitude', 'direction'])) )\
            and p in direction_dict:
                direction_str = direction_dict[p]
            elif  p in sign_dict:
                sign_str = sign_dict[p]
            else:
                prefix.append(p)
    
        feat = '_'.join(prefix) + direction_str + sign_str
        
        
        if not 'upsilon_turns' in feat:
            if 'upsilon_turn' in feat:
                feat = feat.replace('upsilon_turn', 'upsilon_turns')
            elif 'upsilon' in feat:
                feat = feat.replace('upsilon', 'upsilon_turns')
        
        if not 'omega_turns' in feat:
            if 'omega_turn' in feat:
                feat = feat.replace('omega_turn', 'omega_turns')
            elif 'omega' in feat:
                feat = feat.replace('omega', 'omega_turns')
            
        feat = feat.replace('-', '_')
        feat = feat.replace('coil_', 'coils_')
        feat = feat.replace('forward_motion', 'forward')
        feat = feat.replace('paused_motion', 'paused')
        feat = feat.replace('backward_motion', 'backward')
        
        new_names.append(feat)
        
    #%%
    assert not [x for x in new_names if not x in df.columns]
    mean_old.columns = new_names
    feat_old = mean_old[good_feats['feat']]
    
    feat_mean = df.groupby('strain').agg('mean')
    feat_mean = feat_mean[good_feats['feat']]
    
    
    #%%
    def _remove_bb(x):
        if 'backcrossed' in x:
            x = x.split('(backcrossed')[0]
        return x
    
    sql = '''
    SELECT name, description
    FROM strains
    '''
    
    strain_names = pd.read_sql(sql, con=conn)
    strain_names['description'] = strain_names['description'].apply(_remove_bb)
    
    strain_dict = {x['description']:x['name'] for ii, x in strain_names.iterrows()}
    #'osm-9(ky10)trpa-1(ok999)IV':'osm-9(ky10); trpa-1(ok999)IV'
    
    valid_rows = [x for x in mean_old.index if x in strain_dict]
    feat_old_n = feat_old.loc[valid_rows]
    feat_old_n.index = [strain_dict[x] for x in feat_old_n.index]
    
    
    feat_mean_n = feat_mean.loc[feat_old_n.index]
    #%%
    ratio_data =[(feat, (feat_mean_n[feat].abs()-feat_old_n[feat].abs())/feat_old_n[feat].abs())  for feat in feat_mean_n]
   
    columns, dat = zip(*ratio_data)
    
    df_ratio = pd.concat(dat, axis=1)
    df_ratio.columns = columns
    
    #%%
    
    weird_d = df_ratio.abs()>0.1
    weird_d.sum().sort_values().plot()
    #%%
    
    ss = 'N2'
    d1 = df_ratio.loc[ss].abs()
    d2 = feat_old_n.loc[ss]
    d3 = feat_mean_n.loc[ss]
    #%%
    g = [x for x in d2.index if 'eigen' in x]
    dd = pd.concat((d2[g], d3[g], d1[g]), axis=1)
    
    #%%
    bad_parts = ['_distance', '_time', 'motion_direction']
    good_feats = [x for x in df_ratio.columns if not any(f in x for f in bad_parts)]
    
    
    