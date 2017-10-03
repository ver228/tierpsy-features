#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:45:47 2017

@author: ajaver
"""

import os
import itertools
import pandas as pd
import numpy as np
from functools import partial

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, SparsePCA, FastICA
from matplotlib.backends.backend_pdf import PdfPages

import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats import f_oneway

#from tierpsy_features.features import timeseries_columns, ventral_signed_columns, morphology_columns

#%%
def _anova_analysis(feat_means_df, feats2check, pairs2compare = None):
    feat_strain_g = feat_means_df.groupby('strain')
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
    feat, fstats, pvalue, df1, df2 = zip(*stats)
    fstatistics = pd.DataFrame(list(zip(fstats, pvalue, df1, df2)), 
                               index=feat, columns=['fstat', 'pvalue', 'df1', 'df2'])
    
    reject, pvals_corrected, alphacSidak, alphacBonf = \
                smm.multipletests(fstatistics['pvalue'].values, method = 'fdr_tsbky')
    fstatistics['pvalue_adj'] = pvals_corrected
    
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
                               columns=['strain1', 'strain2', 'feat', 'tstatistic', 'pvalue'])
    
    reject, pvals_corrected, alphacSidak, alphacBonf = \
                smm.multipletests(multi_comparisons['pvalue'].values, method = 'fdr_tsbky')
    multi_comparisons['pvalue_adj'] = pvals_corrected

    return fstatistics, multi_comparisons
#%%
if __name__ == '__main__':
    save_dir_root = './anova_tests'
    
    comparsions_keys = dict(
     fm6 = (
            ['DAG356', 'DAG515', 'DAG618'], 
            [('DAG356', 'DAG515'), ('DAG356', 'DAG618'), ('DAG515', 'DAG618')]
            ),
     rescue = (
             ['DAG356', 'DAG618', 'DAG658', 'DAG680'],
             [('DAG356', 'DAG618'), ('DAG356', 'DAG658'),
              ('DAG356', 'DAG680'), ('DAG658', 'DAG680')]
             ),
     
     splicing = (
             ['N2', 'DAG677', 'DAG678', 'DAG679'],
             [('N2', 'DAG677'), ('N2', 'DAG678'),
              ('N2', 'DAG679')]
             ),
      promoter = (
             ['N2', 'DAG666', 'DAG667', 'DAG668', 'DAG676'],
             [('N2', 'DAG666'), ('N2', 'DAG667'),
              ('N2', 'DAG668'), ('N2', 'DAG676')]
             )
      )
    
    
    feat_means_df = pd.read_csv('swiss_strains_stats.csv', index_col=False)
    index_cols = ['strain']
    feats2check = [x for x in feat_means_df if x not in index_cols]
    feats2check = [x for x in feats2check if 'IQR' not in x]
    for comparison_type, (strain_order, pairs2compare) in comparsions_keys.items():
        save_dir = os.path.join(save_dir_root, comparison_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(comparison_type)
        df = feat_means_df[feat_means_df['strain'].isin(strain_order)]
        fstatistics, multi_comparisons = _anova_analysis(df, feats2check, pairs2compare = None)
        fstatistics = fstatistics.sort_values(by='pvalue_adj')

        #%%
        
        save_name = os.path.join(save_dir, 'boxplot.pdf')
        with PdfPages(save_name) as pdf_pages:
            for feat, row in fstatistics.iterrows():
                f, ax = plt.subplots(figsize=(15, 6))
                sns.boxplot(x ='strain', 
                            y = feat, 
                            data = feat_means_df, 
                            order= strain_order
                            )
                sns.swarmplot(x ='strain', y = feat, data = feat_means_df, color=".3", linewidth=0, order= strain_order)
                ax.xaxis.grid(True)
                ax.set(ylabel="")
                sns.despine(trim=True, left=True)
                
                args = (feat, int(row['df1']), int(row['df2']), row['fstat'], row['pvalue'])
                strT = '{} | f({},{})={:.3} | (p-value {:.3})'.format(*args)
                plt.suptitle(strT)
                plt.xlabel('')
                
                pdf_pages.savefig()
                plt.close()
        #%%
        dd = {x:ii for ii,x in enumerate(fstatistics.index)}
        multi_comparisons['m'] = multi_comparisons['feat'].map(dd)
        multi_comparisons = multi_comparisons.sort_values(by='m')
        del multi_comparisons['m']
        
        save_name = os.path.join(save_dir, 'pair_comparisons.csv')
        multi_comparisons.to_csv(save_name, index=False)
#%%      


    
#    #%%
#    strain_order = ['N2', 'DAG356', 
#                    'TR2171', 'DAG618', 
#                    'DAG658', 'DAG680', 
#                    'DAG515', 'DAG675', 
#                    'DAG666', 'DAG667', 'DAG668', 'DAG676', 
#                    'DAG677', 'DAG678', 'DAG679']
#    cols_ind = [0, 0, 
#                1, 1, 
#                2, 2, 
#                3, 3, 
#                4, 4, 4, 4, 4, 4, 4]
#    
#    
#    current_palette = sns.color_palette()
#    cols = [current_palette[x] for x in cols_ind]
#    assert len(strain_order) == len(cols)
#    col_dict = {k:v for k,v in zip(strain_order, cols)}
#    
#    
#    
#    #%%
#    if True:
#        #median_values = feat_strain_g.median()
#        
#        save_name =  os.path.join('{}/boxplot_anova.pdf'.format(save_dir))
#        with PdfPages(save_name) as pdf_pages:
#            for feat, row in stat_values.iterrows():
#                
#                f, ax = plt.subplots(figsize=(15, 6))
#                sns.boxplot(x ='strain', 
#                            y = feat, 
#                            data = feat_means_df, 
#                            order= strain_order,
#                            palette = col_dict
#                            )
#                sns.swarmplot(x ='strain', y = feat, data = feat_means_df, color=".3", linewidth=0, order= strain_order)
#                ax.xaxis.grid(True)
#                ax.set(ylabel="")
#                sns.despine(trim=True, left=True)
#                plt.suptitle('{} (p-value {:.3})'.format(feat, row['pvalue']))
#                plt.xlabel('')
#                
#                #dd = median_values[feat].argsort()
#                #strain_order = list(dd.index[dd.values])
#                #n2_ind = strain_order.index('N2')
#                #plt.plot((n2_ind,n2_ind), plt.ylim(), '--k', linewidth=2)
#                
#                pdf_pages.savefig()
#                plt.close(f)
#    
#    #%%
#    df = feat_means_df.dropna()
#    
#    valid_feats = [x for x in feat_means_df if x not in ['strain'] ]
#    X = df[valid_feats].values.copy()
#    
#    x_min, x_max = np.min(X, 0), np.max(X, 0)
#    X = (X - x_min)/(x_max - x_min)
#    
#    #X = (X - np.mean(X, 0))/np.std(X, 0)
#    #%%
#    pca = PCA()
#    X_pca = pca.fit_transform(X)
#    
#    pca_s = SparsePCA()
#    X_pca_s = pca_s.fit_transform(X)
#    #%%
#    
#    #for p in [5, 8, 10, 12, 15, 20, 30]:
#    tsne = TSNE(n_components=2, 
#                #perplexity = p,
#                init='pca',
#                verbose=1, 
#                n_iter=10000
#                )# random_state=0)
#    X_tsne = tsne.fit_transform(X)
#    
#    plt.figure()
#    plt.plot(X_tsne[:, 0], X_tsne[:, 1], 'o')
#    #%%
#    from matplotlib.lines import Line2D
#    import itertools
#    
#    marker_cycle = itertools.cycle(Line2D.filled_markers)
#    mks = [next(marker_cycle) for _ in strain_order]
#    #%%
#    save_name =  os.path.join('{}/clustering.pdf'.format(save_dir))
#    with PdfPages(save_name) as pdf_pages:
#        
#        dat = {'t-SNE':X_tsne, 'PCA':X_pca, 'PCA_Sparse':X_pca_s}
#        #%%
#        for k,Xp in dat.items():
#            
#            
#            X_df = pd.DataFrame(Xp[:, 0:2], columns=['X1', 'X2'])
#            X_df['strain'] = df['strain'].values
#            
#            g = sns.lmplot('X1', # Horizontal axis
#               'X2', # Vertical axis
#               data=X_df, # Data source
#               fit_reg=False, # Don't fix a regression line
#               hue = 'strain',
#               hue_order = strain_order,
#               palette = col_dict,
#               size= 8,
#               scatter_kws={"s": 100},
#               legend=False,
#               aspect = 1.2,
#               markers = mks
#               )
#            
#            box = g.ax.get_position() # get position of figure
#            g.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position
#    
#            g.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#            plt.title(k)
#            pdf_pages.savefig()
#            
#        plt.close('all')        
#    #%%
#    
#    strT = 'PCA'
#    Xr = X_pca
#    
#    save_name =  os.path.join('{}/{}.pdf'.format(save_dir, strT))
#    with PdfPages(save_name) as pdf_pages:
#    
#        plt.figure()
#        plt.plot(np.cumsum(pca.explained_variance_ratio_), '.')
#        plt.title('Explained Variance')
#        pdf_pages.savefig()
#        
#        for n in range(10):
#            df['p_dist'] = Xr[:, n]
#            
#            f, ax = plt.subplots(figsize=(15, 6))
#            
#            sns.boxplot(x = 'strain', 
#                        y = 'p_dist', 
#                        data = df, 
#                        order= strain_order,
#                        palette = col_dict
#                        )
#            sns.swarmplot(x = 'strain', 
#                          y = 'p_dist', 
#                          data = df,
#                          color=".3", 
#                          linewidth=0, 
#                          order= strain_order
#                          )
#            
#            plt.title('{}_{} var explained: {:.2}'.format(strT, n+1, pca.explained_variance_ratio_[n]))
#            pdf_pages.savefig()
#            
#        plt.close('all')
#    #%%
#    
#    strT = 'PCA_Sparse'
#    Xr = X_pca_s
#    
#    save_name =  os.path.join('{}/{}.pdf'.format(save_dir, strT))
#    with PdfPages(save_name) as pdf_pages:
#        #http://www.tandfonline.com/doi/pdf/10.1198/106186006X113430?needAccess=true
#        q, r = np.linalg.qr(X_pca_s)
#        explained_variance = np.diag(r)**2
#        explained_variance_ratio = explained_variance/np.sum(explained_variance)
#        
#        plt.figure()
#        plt.plot(np.cumsum(explained_variance_ratio), '.')
#        plt.title('Explained Variance')
#        pdf_pages.savefig()
#        
#        
#        for n in range(10):
#            df['p_dist'] = Xr[:, n]
#            #df['p_dist'] = np.linalg.norm((X_pca - n2_m)[:,:(n+1)], axis=1)
#            
#            f, ax = plt.subplots(figsize=(15, 6))
#            
#            sns.boxplot(x = 'strain', 
#                        y = 'p_dist', 
#                        data = df, 
#                        order= strain_order,
#                        palette = col_dict
#                        )
#            sns.swarmplot(x = 'strain', 
#                          y = 'p_dist', 
#                          data = df,
#                          color=".3", 
#                          linewidth=0, 
#                          order= strain_order
#                          )
#            
#            pca_s.components_[0, :]
#            plt.title('{}_{} var explained: {:.2}'.format(strT, n+1, explained_variance_ratio[n]))
#            pdf_pages.savefig()
#        
#        plt.close('all')
#    #%%
#    save_name =  os.path.join('{}/{}.txt'.format(save_dir, strT))
#    with open(save_name, 'w') as fid:
#        for n in range(10):
#            vec = pca_s.components_[n, :]
#            inds, = np.where(vec!=0)
#            dd = [(valid_feats[ii],vec[ii]) for ii in inds]
#            
#            fid.write('***** PCA Sparse {} *****\n'.format(n+1))
#            for feat in sorted(dd):
#                fid.write('{} {:.3}\n'.format(*feat))
#            fid.write('\n')
    