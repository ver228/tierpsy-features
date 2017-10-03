#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:55:02 2017

@author: ajaver
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib.lines import Line2D


matplotlib.style.use('ggplot')
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns

import itertools
from functools import partial
from scipy.stats import f_oneway
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind

from tierpsy_features.features import timeseries_columns, ventral_signed_columns, morphology_columns

#%%
def _feat_boxplots(feats_data, feats_anova, feats2plot, save_name):
    ##### Box plots
    col_wrap = min(4, feats_data['time_group'].unique().size)
    
    
    with PdfPages(save_name) as pdf_pages:
        for feat in feats2plot:
            
            g = sns.factorplot(x='cohort_n', 
                          y=feat,  
                          data=feats_data,
                          col='time_group',
                          col_wrap = col_wrap,
                          kind="box",
                          legend_out = True,
                          size=5
                         )
            
            anova_g = feats_anova.groupby(('feat', 'time_group'))
            for ax in g.axes:
                strT = ax.get_title()
                time = float(strT.split()[-1])
                dat = anova_g.get_group((feat, time))
                
                pval = dat['pvalue_adj'].values[0]
                dd = ' | pval:{:0.3e}'.format(pval)
                if pval < 0.1:
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
#def _boxplot_pca(labels, p_dist, label_order, col_dict, variance_ratio):
#    df_p = pd.DataFrame()
#    df_p['p_dist'] = p_dist
#    df_p['labels'] = labels
#    
#    f, ax = plt.subplots(figsize=(15, 6))
#    
#    sns.boxplot(x = 'labels', 
#                y = 'p_dist', 
#                data = df_p, 
#                order= label_order,
#                palette = col_dict
#                )
#    sns.swarmplot(x = 'labels', 
#                  y = 'p_dist', 
#                  data = df_p,
#                  color=".3", 
#                  linewidth=0, 
#                  order= label_order
#                  )
#    
#    plt.title('{}_{} var explained: {:.2}'.format(k, n+1, variance_ratio))
#%%
def get_fstatistics(feat_means_df, key2group, feats2check):
    feat_strain_g = feat_means_df.groupby(key2group)
    #anova test
    stats = []
    for feat in feats2check:
        dat = [g[feat].dropna().values for _, g in feat_strain_g]
        fstats, pvalue = f_oneway(*dat)

        #get the degree's of freedom. I got this formulas from the f_oneway scipy repo
        #prob = special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
        bign = sum(x.size for x in dat)
        num_groups = len(dat)
        dfbn = num_groups - 1
        dfwn = bign - num_groups
        
        stats.append((feat, fstats, pvalue, dfbn, dfwn))
        
    fstatistics = pd.DataFrame(stats, columns=['feat', 'fstat', 'pvalue', 'df1', 'df2'])
    return fstatistics


def get_pairwise_comparison(feat_means_df, key2group, feats2check, pairs2compare = None):
    feat_strain_g = feat_means_df.groupby(key2group)
    #pairwise comparsion
    all_comparisons = []
    if pairs2compare is None:
        pairs2compare = list(itertools.combinations(feat_strain_g.groups.keys(), 2))
    
    for x1,x2 in pairs2compare:
        a = feat_strain_g.get_group(x1).dropna()
        b = feat_strain_g.get_group(x2).dropna()
    
        for feat in feats2check:
            tstatistic, pvalue = ttest_ind(a[feat].values, b[feat].values)
            all_comparisons.append((x1, x2, feat, tstatistic, pvalue))
    
    multi_comparisons = pd.DataFrame(all_comparisons, 
                               columns=[key2group + '1', key2group + '2', 'feat', 'tstatistic', 'pvalue'])
    
    return multi_comparisons

def add_pvalue_adj(df):
    #I am repeating this step but i need to consider all the comparsions to get the correct value
    reject, pvals_corrected, alphacSidak, alphacBonf = \
                smm.multipletests(df['pvalue'].values, method = 'fdr_tsbky')
    df['pvalue_adj'] = pvals_corrected
    
    return df
def _cluster_analysis(feats_data):
    #%% ##### Clustering analysis
    df_n = feats_data.dropna()
    df = df_n[set_feats]
    index_data = df_n[index_cols].reset_index()
    
    X = df.values.copy()
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min)/(x_max - x_min)
    #%% #### labels and indexes vectors
    format_func = lambda row : 'C{}_T{}'.format(int(row['cohort_n']), ('%1.1f' % row['time_group']).zfill(4)) 
    labels = [format_func(row) for _, row in index_data.iterrows()]
    
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
    
    
            
    dat = {'t-SNE':X_tsne, 'PCA':X_pca, 'PCA_Sparse':X_pca_s}
    save_name = '{}/clustering_{}.pdf'.format(save_dir, feat_prefix)
    with PdfPages(save_name) as pdf_pages:
        
        for k, Xp in dat.items():
            _plot_clusters(Xp, labels, label_order, col_dict, mks)
            plt.title(k)
            pdf_pages.savefig()
            plt.close()
    return dat
#%%
if __name__ == '__main__':
    for set_prefix in ['Vortex', '4h']:
        save_dir_root = '/Users/ajaver/OneDrive - Imperial College London/development/031017/Set_{}'
        feats_data = pd.read_csv(set_prefix + '_data.csv')
        
        index_cols = ['exp_n', 'cohort_n', 'time_group']
        valid_feats = [x for x in feats_data if x not in index_cols]
        morphology_feats = [x for x in valid_feats if any(f in x for f in morphology_columns)]
        no_morphology_feats = [x for x in valid_feats if not x in morphology_feats]
        
        all_dat = []
        for time_group, dat in feats_data.groupby('time_group'):
            fstats = get_fstatistics(dat, 'cohort_n', valid_feats)
            fstats.insert(0, 'time_group', time_group)
            
            multi_comp = get_pairwise_comparison(dat, 'cohort_n', valid_feats)
            multi_comp.insert(0, 'time_group', time_group)
            
            
            all_dat.append((fstats, multi_comp))
        
        #pack data into dataframes
        dd = map(partial(pd.concat, ignore_index = True), zip(*all_dat))
        #correct for multiple comparsions
        fstatistics, multi_comparisons = map(add_pvalue_adj, dd)
        
        #I am repeating this step but i need to consider all the comparsions to get the correct value
        reject, pvals_corrected, alphacSidak, alphacBonf = \
                    smm.multipletests(multi_comparisons['pvalue'].values, method = 'fdr_tsbky')
        multi_comparisons['pvalue_adj'] = pvals_corrected
        
        #order the features from more significative to less
        feat_order = fstatistics.sort_values(by='pvalue_adj')['feat'].drop_duplicates(keep='first').values
        
        
        feat_type_group = [(no_morphology_feats, 'NO_morph'), 
                             (morphology_feats, 'morph'),
                             (valid_feats, 'all')
                             ]
        
        
        #%%
        save_dir = save_dir_root.format(set_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for set_feats, feat_prefix in feat_type_group:
            save_name = '{}/boxplot_{}.pdf'.format(save_dir, feat_prefix)
            feats2plot = [x for x in feat_order if x in set_feats]
            _feat_boxplots(feats_data, fstatistics, feats2plot, save_name)
            
            _cluster_analysis(feats_data)
        
        #%%
#        ##### PCA's
#        for k, Xp in [('PCA', X_pca), ('PCA_Sparse', X_pca_s)]:
#            
#            
#            if k == 'PCA':
#                explained_variance_ratio = pca.explained_variance_ratio_
#            
#            elif k == 'PCA_Sparse':
#                #http://www.tandfonline.com/doi/pdf/10.1198/106186006X113430?needAccess=true
#                q, r = np.linalg.qr(X_pca_s)
#                explained_variance = np.diag(r)**2
#                explained_variance_ratio = explained_variance/np.sum(explained_variance)
#                    
#            save_name = '{}/{}_{}.pdf'.format(save_dir, k, feat_prefix)
#            with PdfPages(save_name) as pdf_pages:
#                
#                plt.figure()
#                plt.plot(np.cumsum(explained_variance_ratio), '.')
#                plt.title('Explained Variance')
#                pdf_pages.savefig()
#                
#                for n in range(10):
#                    _boxplot_pca(labels, Xp[:, n], label_order, col_dict, explained_variance_ratio[n])
#                    pdf_pages.savefig()
#                
#                plt.close('all')