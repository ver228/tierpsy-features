#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:55:02 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import glob
import os

from tierpsy.helper.params import read_fps

import statsmodels.stats.multitest as smm

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib.lines import Line2D


matplotlib.style.use('ggplot')
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, SparsePCA

import seaborn as sns
from scipy.stats import f_oneway

from tierpsy_features.features import timeseries_columns, ventral_signed_columns, morphology_columns
#%%
def get_df_quantiles(df):
    q_vals = [0.1, 0.5, 0.9]
    iqr_limits = [0.25, 0.75]
    
    valid_q = q_vals + iqr_limits
    Q = df[ventral_signed_columns].abs().quantile(valid_q)
    Q.columns = [x+'_abs' for x in Q.columns]
    
    vv = [x for x in timeseries_columns if not x in ventral_signed_columns]
    Q_s = df[vv].quantile(valid_q)
    feat_mean = pd.concat((Q, Q_s), axis=1)
    
    dat = []
    for q in q_vals:
        q_dat = feat_mean.loc[q]
        q_str = '_{}th'.format(int(round(q*100)))
        for feat, val in q_dat.iteritems():
            dat.append((val, feat+q_str))
    
    
    IQR = feat_mean.loc[0.75] - feat_mean.loc[0.25]
    for feat, val in IQR.iteritems():
        dat.append((val, feat + '_IQR'))
    feat_mean_s = pd.Series(*list(zip(*dat)))
    return feat_mean_s
#%%
def averages_by_time(df, fps, window_minutes = 5):
    window_frames = int(60*window_minutes*fps)
    df['timestamp_m'] = np.floor(df['timestamp']/window_frames).astype(np.int)
    dat_agg = {tt : get_df_quantiles(dat) for tt, dat in df.groupby('timestamp_m')}
    dat_agg =  pd.concat(dat_agg, axis=1)
    return dat_agg
#%%
def _feat_boxplots(feats_data, feats_anova, feats2plot, save_name):
    ##### Box plots
    col_wrap = min(4, feats_data['time'].unique().size)
    
    
    with PdfPages(save_name) as pdf_pages:
        for feat in feats2plot:
            anova_g = feats_anova.groupby(('feat', 'time'))
            g = sns.factorplot(x='exp_n', 
                          y=feat, 
                          hue='cohort_n', 
                          data=feats_data,
                          col='time',
                          col_wrap = col_wrap,
                          kind="box",
                          legend_out = True,
                          size=5
                         )
            
            for ax in g.axes:
                strT = ax.get_title()
                time = float(strT.split()[-1])
                dat = anova_g.get_group((feat, time))
                for _, row in dat.sort_values(by='exp_n').iterrows():
                    dd = ' | Exp{} pval:{:0.3e}'.format(row['exp_n'],row['pval_corrected'])
                    if row['pval_corrected'] < 0.1:
                        dd += '*'
                    strT += dd
                    
                ax.set_title(strT)
            pdf_pages.savefig()
            plt.close()
#%%
def  _plot_clusters(Xp, labels, label_order, col_dict, mks):
    X_df = pd.DataFrame(Xp[:, 0:2], columns=['X1', 'X2'])
    X_df['labels'] = labels
    
    g = sns.lmplot('X1', # Horizontal axis
       'X2', # Vertical axis
       data=X_df, # Data source
       fit_reg=False, # Don't fix a regression line
       hue = 'labels',
       hue_order = label_order,
       palette = col_dict,
       size= 8,
       scatter_kws={"s": 100},
       legend = False,
       aspect = 1.2,
       markers = mks
       )
    box = g.ax.get_position() # get position of figure
    g.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position

    g.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#%%
def _boxplot_pca(labels, p_dist, label_order, col_dict, variance_ratio):
    df_p = pd.DataFrame()
    df_p['p_dist'] = p_dist
    df_p['labels'] = labels
    
    f, ax = plt.subplots(figsize=(15, 6))
    
    sns.boxplot(x = 'labels', 
                y = 'p_dist', 
                data = df_p, 
                order= label_order,
                palette = col_dict
                )
    sns.swarmplot(x = 'labels', 
                  y = 'p_dist', 
                  data = df_p,
                  color=".3", 
                  linewidth=0, 
                  order= label_order
                  )
    
    plt.title('{}_{} var explained: {:.2}'.format(k, n+1, variance_ratio))

#%%
if __name__ == '__main__':
    window_minutes = 15
    #fname = '/Volumes/behavgenom_archive$/Solveig/All/Results/Experiment1/170713_deve_1/deve_1_day1_Set0_Pos0_Ch1_13072017_140054_featuresN.hdf5'
    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    
    
    set_prefix = '4h'
    exp_valid = [7,8]
    window_minutes = 15
    
    #set_prefix = 'Vortex'
    #exp_valid = [5,6]
    #window_minutes = 7.5
    
    is_normalize_data = False

    if is_normalize_data:
        set_prefix += '_N'
    
    fnames = glob.glob(os.path.join(dname, '**', '*_featuresN.hdf5'), recursive = True)
    fnames = [x for x in fnames if int(x.split('Experiment')[-1].split('/')[0]) in exp_valid]
    #%%
    stat_data = {}
    stat_data_n = {}
    for ifname, fname in enumerate(fnames):
        print(ifname+1, len(fnames))
        exp_key = fname.split(os.sep)[-2].split('_')[-1]
        
        with pd.HDFStore(fname, 'r') as fid:
            if '/provenance_tracking/FEAT_TIERPSY' in fid:
                timeseries_features = fid['/timeseries_features']
                fps = read_fps(fname)
                
                
                if 'Vortex' in set_prefix:
                    # I want to takeout data one minute before and after she vortex the worms (it might be too noisy)
                    lower = 14*fps*60
                    higher = 16*fps*60 
                    good = (timeseries_features['timestamp']<lower) | (timeseries_features['timestamp']>higher)
                    timeseries_features = timeseries_features[good]
                    
                    timeseries_features = timeseries_features[timeseries_features['timestamp']<= 29.8*fps*60]
                else:
                    timeseries_features = timeseries_features[timeseries_features['timestamp']<= 119.8*fps*60]
                
                    
            else:
                continue
            
        if not exp_key in stat_data:
            stat_data[exp_key] = []
            stat_data_n[exp_key] = []
            
        if is_normalize_data:
        
            median_length = timeseries_features.groupby('worm_index').agg({'length':'median'})
            median_length_vec = timeseries_features['worm_index'].map(median_length['length'])
            
            feats2norm = [
                   'speed',
                   'relative_speed_midbody', 
                   'relative_radial_velocity_head_tip',
                   'relative_radial_velocity_neck',
                   'relative_radial_velocity_hips',
                   'relative_radial_velocity_tail_tip',
                   'head_tail_distance',
                   'major_axis', 
                   'minor_axis', 
                   'dist_from_food_edge'
                   ]
        
            for f in feats2norm:
                if f in timeseries_features:
                    timeseries_features[f] /= median_length_vec
            
            curv_feats = ['curvature_head',
                           'curvature_hips', 
                           'curvature_midbody', 
                           'curvature_neck',
                           'curvature_tail']
            for f in curv_feats:
                if f in timeseries_features:
                    timeseries_features[f] *= median_length_vec
        
        
        dat = averages_by_time(timeseries_features, fps, window_minutes)
        stat_data[exp_key].append(dat)
        
    
    #%%
    #put all the data into a nice dataframe
    
    exp_ns = set(int(x.split('co')[0][3:]) for x in stat_data.keys())
    cohort_ns = set(int(x.split('co')[-1]) for x in stat_data.keys())
    #rearrange the data
    data = []
    for exp_n in exp_ns:
        for cohort_n in cohort_ns:
            exp_key = 'exp{}co{}'.format(exp_n, cohort_n)
            
            for dat in stat_data[exp_key]:
                for ind_t in dat.columns:
                    dd = dat[ind_t]
                    dd['time'] = ind_t*window_minutes
                    dd['exp_n'] = exp_n
                    dd['cohort_n'] = cohort_n
                    data.append(dd)
    #%%
    df = pd.concat(data, axis=1, ignore_index=True).T
    df['exp_n'] = df['exp_n'].astype(np.int)
    df['cohort_n'] = df['cohort_n'].astype(np.int)
    
    
    #let's do a one way anova with the different (exp_n,time) combinations
    ftest_stats = []
    index_cols = ['time', 'exp_n', 'cohort_n']
    for (exp_n, time), dat in df.groupby(('exp_n', 'time')):
        dat_g = dat.groupby('cohort_n')
        for feat in df:
            if not feat in index_cols:
                dat = [g[feat].dropna().values for _, g in dat_g]
                fstats, pvalue = f_oneway(*dat)
                ftest_stats.append((exp_n, time, feat, fstats, pvalue))
    
    anova_df = pd.DataFrame(ftest_stats, columns=['exp_n', 'time', 'feat', 'fstat', 'pvalue'])
    
    #%%
    #since i know that the morphology among the cohorts is different I treat it as a separated dataset
    index_cols = ['time', 'exp_n', 'cohort_n']
    valid_feats = [x for x in df if x not in index_cols]
    morphology_feats = [x for x in valid_feats if any(f in x for f in morphology_columns)]
    no_morphology_feats = [x for x in valid_feats if not x in morphology_feats]
    
    #%%
    
    
    feats_anova = anova_df[anova_df['exp_n'].isin(exp_valid)].copy()
    feats_data = df[df['exp_n'].isin(exp_valid)]
    
    #Get the multiple hypothesis comparison
    reject, pvals_corrected, alphacSidak, alphacBonf = \
                smm.multipletests(feats_anova['pvalue'].values, method = 'fdr_tsbky')
    feats_anova['pval_corrected'] = pvals_corrected
    
    #Get the feature order by getting the smaller set of p values for a given experiment and time
    dd = [(feat, dat['pvalue'].sum()) for (_, feat), dat in feats_anova.groupby(('time', 'feat'))]
    dd = pd.DataFrame(sorted(dd, key = lambda x : x[1]), columns=['feat', 'combined_pvalues'])
    feat_order = dd.sort_values(by='combined_pvalues')['feat'].drop_duplicates(keep='first').values
    
    #%%
    save_dir = '/Users/ajaver/OneDrive - Imperial College London/development/290917/Set_{}'.format(set_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for set_feats, feat_prefix in [(no_morphology_feats, 'NO_morph'), 
                                 (morphology_feats, 'morph'),
                                 (valid_feats, 'all')]:
        
        save_name = '{}/boxplot_{}.pdf'.format(save_dir, feat_prefix)
        feats2plot = [x for x in feat_order if x in set_feats]
        _feat_boxplots(feats_data, feats_anova, feats2plot, save_name)
        #%%
        ##### Clustering analysis
        
        df_n = feats_data.dropna()
        df = df_n[set_feats]
        index_data = df_n[index_cols].reset_index()
        
        X = df.values.copy()
        
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min)/(x_max - x_min)
        #%%
        #### labels and indexes vectors
        labels = ['C{}_T{}'.format(int(row['cohort_n']), row['time']) for _, row in index_data.iterrows()]
        
        current_palette = sns.color_palette()
        cols = [current_palette[x-1] for x in index_data['cohort_n'].values]
        col_dict = {k : v for k,v in zip(labels, cols)}
        label_order = sorted(list(set(labels)))
        
        dd = [x.split('_T')[-1] for x in label_order]
        dd_dict = {x : Line2D.filled_markers[ii] for ii, x in enumerate(set(dd))}
        mks = [dd_dict[x] for x in dd]
        
        tsne = TSNE(n_components=2, 
                        #perplexity = p,
                        init='pca',
                        verbose=1, 
                        n_iter=10000
                        )
        X_tsne = tsne.fit_transform(X)
        
        pca_s = SparsePCA()
        X_pca_s = pca_s.fit_transform(X)
        
        pca = PCA()
        X_pca = pca.fit_transform(X)
        #%%
        
        dat = {'t-SNE':X_tsne, 'PCA':X_pca, 'PCA_Sparse':X_pca_s}
        save_name = '{}/clustering_{}.pdf'.format(save_dir, feat_prefix)
        with PdfPages(save_name) as pdf_pages:
            
            for k, Xp in dat.items():
                _plot_clusters(Xp, labels, label_order, col_dict, mks)
                plt.title(k)
                pdf_pages.savefig()
                plt.close()
        #%%
        ##### PCA's
        for k, Xp in [('PCA', X_pca), ('PCA_Sparse', X_pca_s)]:
            
            
            if k == 'PCA':
                explained_variance_ratio = pca.explained_variance_ratio_
            
            elif k == 'PCA_Sparse':
                #http://www.tandfonline.com/doi/pdf/10.1198/106186006X113430?needAccess=true
                q, r = np.linalg.qr(X_pca_s)
                explained_variance = np.diag(r)**2
                explained_variance_ratio = explained_variance/np.sum(explained_variance)
                    
            save_name = '{}/{}_{}.pdf'.format(save_dir, k, feat_prefix)
            with PdfPages(save_name) as pdf_pages:
                
                plt.figure()
                plt.plot(np.cumsum(explained_variance_ratio), '.')
                plt.title('Explained Variance')
                pdf_pages.savefig()
                
                for n in range(10):
                    _boxplot_pca(labels, Xp[:, n], label_order, col_dict, explained_variance_ratio[n])
                    pdf_pages.savefig()
                
                plt.close('all')