import pickle
import numpy as np
from reader import read_feats
from RFE_simple_reduced import feats2remove

if __name__ == "__main__":
    feat_data, col2ignore_r = read_feats() 
    #%%
    save_name = 'RFE_SoftMax_F1.pkl'
    #save_name =  'RFE_SoftMax_F1_reduced.pkl'#'RFE_SoftMax_F1.pkl'
    #save_name =  'RFE_SoftMax_Flog2_reduced.pkl'
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    #%%
    df = feat_data['tierpsy']
    v_cols = [x for x in df.columns if not 'hu' in x]
    v_cols = [x for x in v_cols if not any(f in x for f in feats2remove)]
    
    feat_data['tierpsy_reduced'] = df[v_cols]
    
    
    del feat_data['all']
    del feat_data['OW']
    
    #%%
    res_db = {}
    for (db_name, i_fold), dat in results:
        if db_name not in res_db:
            res_db[db_name] = []
            
        feats, vals = zip(*dat)
        loss, acc, f1 = map(np.array, zip(*vals))
        
        feats = sum(feats , [])
        res_db[db_name].append((feats, loss, acc, f1))
        
    
    #%% Check I have all the available features
    from collections import Counter
    
    search_window_top = 8
    
    top_f_dict = {}
    for db_name, db_data in res_db.items():
        if not db_name in feat_data:
            print('???', db_name)
            continue
        
        all_feats = [x for x in feat_data[db_name] if x not in col2ignore_r]
        
        collect_top_feats = []
        for (feats, loss, acc, f1) in db_data:
            missing_feats = set(feats) - set(all_feats)
            assert not missing_feats
            
            collect_top_feats += feats[-search_window_top:]
        
        
        dd = Counter(collect_top_feats)
        dd = [(v,k) for k,v in sorted(dd.items(), key = lambda x : x[1])]
        top_f_dict[db_name] = dd
    #%%
    import pandas as pd
    from scipy.stats import ttest_ind, ranksums
    
    df = feat_data['tierpsy']
    #df = df[['date', 'strain']+ maybe_good].copy()
    
    df['date'] = pd.to_datetime(df['date'])
    
    df_g = df.groupby('strain')
    
    N2_data = df_g.get_group('N2')
    
    time_offset_allowed = 7
    
    valid_feats = [x for x in df if x not in col2ignore_r]
    
    ctr_sizes = []
    all_pvalues = []
    for ss, s_data in df_g:
        print(ss)
        if ss == 'N2':
            continue
        
        offset = pd.to_timedelta(time_offset_allowed, unit='day')
        #ini_date = s_data['date'].min() - offset
        #fin_date = s_data['date'].max() + offset
        
        udates = s_data['date'].map(lambda t: t.date()).unique()
        udates = [pd.to_datetime(x) for x in udates]
        
        good = (N2_data['date'] > udates[0] - offset) & (N2_data['date'] < udates[0] + offset)
        for ud in udates:
            good |= (N2_data['date'] > ud - offset) & (N2_data['date'] < ud + offset)
        ctrl_data = N2_data[good]
        
        ctr_sizes.append((ss, len(ctrl_data)))
        
        
        for ff in valid_feats:
            ctr = ctrl_data[ff].values
            atdf_pvaluesr = s_data[ff].values
            
            #_, p = ttest_ind(ctr, atr)
            _, p = ranksums(ctr, atr)
            assert isinstance(p, float)
            all_pvalues.append((ss, ff, p))
    #%%
    df_pvalues = pd.DataFrame(all_pvalues, columns = ['strain', 'feature', 'pvalue'])
    df_pvalues = df_pvalues.pivot('strain', 'feature', 'pvalue')
    #%%
    pp = df_pvalues*(df_pvalues.shape[0]*df_pvalues.shape[1]) #bonferroni correction
    n_significative = ((pp<0.05).sum(axis = 0)).sort_values()
    n_strains = ((pp<0.05).sum(axis = 1)).sort_values()
    #(pp<0.05).sum(axis=0)
    
    #%%
    p1 = pp.idxmin(axis=1)
    top_per_strain = p1.value_counts()
    #%%
    for ss in pp.T:
        pass
    
        #%%
    [(3, 'curvature_tail_norm_IQR'),
  (3, 'd_eigen_projection_4_w_forward_90th'),
  (3, 'blob_compactness_50th'),
  (3, 'motion_mode_backward_fraction'),
  (3, 'blob_solidity_w_forward_50th'),
  (3, 'curvature_midbody_w_forward_IQR'),
  (3, 'd_length_10th'),
  (3, 'curvature_head_w_backward_IQR'),
  (3, 'width_midbody_norm_10th'),
  (3, 'd_curvature_midbody_w_backward_90th'),
  (3, 'curvature_head_norm_IQR'),
  (3, 'd_relative_radial_velocity_head_tip_w_forward_90th'),
  (4, 'd_relative_speed_midbody_10th'),
  (4, 'blob_quirkiness_w_forward_50th'),
  (4, 'd_eigen_projection_6_IQR'),
  (4, 'motion_mode_paused_fraction'),
  (4, 'd_length_50th'),
  (4, 'length_90th'),
  (4, 'curvature_head_norm_abs_90th'),
  (4, 'width_head_base_norm_50th'),
  (5, 'd_width_head_base_IQR'),
  (5, 'relative_radial_velocity_tail_tip_w_backward_10th'),
  (5, 'width_tail_base_w_backward_10th'),
  (5, 'curvature_head_w_forward_IQR'),
  (5, 'd_curvature_head_w_forward_IQR'),
  (5, 'width_head_base_w_forward_10th')]

#'relative_radial_velocity_head_tip_w_forward_50th'


