#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:23:00 2017

@author: ajaver
"""
import os
import pandas as pd
import numpy as np
import tables
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from compare_ftests import col2ignore

from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pylab as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cmap=plt.cm.Blues
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    np.set_printoptions(precision=2)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%1.2f' % cm[i, j],
                 horizontalalignment="center",
                 fontsize =12,
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




cross_validation_fold = 3
base_strains = ['N2', 'ED3017', 'CX11314', 'LKC34', 
                  'MY16', 'DL238', 'JT11398', 'JU775',
                  'JU258', 'MY23', 'EG4725', 'CB4856'
                  ]
main_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/CeNDR/CeNDR_skel_smoothed.hdf5'
save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/CeNDR/'
feat_files = {
        'OW' : 'ow_features_CeNDR.csv',
        'tierpsy' :'tierpsy_features_CeNDR.csv'
        }

#cross_validation_fold = 10
#base_strains = ['JU393', 'ED3054', 'JU394', 
#                 'N2', 'JU440', 'ED3021', 'ED3017', 
#                 'JU438', 'JU298', 'JU345', 'RC301', 
#                 'AQ2947', 'ED3049',
#                 'LSJ1', 'JU258', 'MY16', 
#                 'CB4852', 'CB4856', 'CB4853',
#                 ]
#main_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/SWDB/SWDB_skel_smoothed_v2.hdf5'
#save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
#feat_files = {
#        'OW_old' : 'ow_features_old_SWDB.csv',
#        'OW' : 'ow_features_SWDB.csv',
#        'tierpsy' :'tierpsy_features_SWDB.csv'
#        }
#%%

with pd.HDFStore(main_file, 'r') as fid:
    df1 = fid['/skeletons_groups']
    df3 = fid['/experiments_data']
    strain_codes = fid['/strains_codes']

cols_to_use = df3.columns.difference(df1.columns)
df3 = df3[cols_to_use]
df3 = df3.rename(columns={'id' : 'experiment_id'})
df = df1.join(df3.set_index('experiment_id'), on='experiment_id')
#%%
valid_indices = {}
with tables.File(main_file, 'r') as fid:
    for set_type in ['train', 'test', 'val']:
        dd = '/index_groups/' + set_type
        if dd in fid:
            v_ind = fid.get_node(dd)[:]
            #use previously calculated indexes to divide data in training, validation and test sets
            valid_indices[set_type] = df.loc[v_ind, 'base_name'].unique()
            
assert not (set(valid_indices['test']) & set(valid_indices['train']))
#%%
sets2test = {}
for db_name, bn in feat_files.items():
    fname = os.path.join(save_dir, 'F_' + bn)
    feats = pd.read_csv(fname)
    
    #s_dict = {s:ii for _, (s,ii) in strain_codes.iterrows()}
    #
    ss = np.sort(feats['strain'].unique())
    s_dict = {s:ii for ii,s in enumerate(ss)}
    feats['strain_id'] = feats['strain'].map(s_dict)
    
    sets2test[db_name] = {}
    for set_type, v_ind in valid_indices.items():
        sets2test[db_name][set_type] = feats[feats['base_name'].isin(v_ind)]


#%%
base_strains_id = {x:i for i,x in enumerate(base_strains)}
classifiers_d = {}
for db_name, datasets  in sets2test.items():
    col_feats = [x for x in datasets['train'].columns if x not in col2ignore]
    
    ss = datasets['train']
    ss = ss[ss['strain'].isin(base_strains)]
    X_train = ss[col_feats].values
    Y_train = ss['strain'].map(base_strains_id).values
    print(db_name, ss.shape)
    
    ss = datasets['test']
    ss = ss[ss['strain'].isin(base_strains)]
    x_test = ss[col_feats].values
    y_test = ss['strain'].map(base_strains_id).values
    print(db_name, ss.shape)
    
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, Y_train)

    proba = clf.predict_proba(x_test)
    top_pred = np.argsort(proba, axis=1)[: ,::-1]
    preds = top_pred==y_test[:, np.newaxis]
    print(db_name, preds[:,0].mean())
    
    classifiers_d[db_name] = clf
    #%%
    dr = {v:k for k,v in base_strains_id.items()}
    
    yt = [dr[x] for x in y_test]
    yp = [dr[x] for x in top_pred[:, 0]]
    labels = sorted(dr.values())
    cm = confusion_matrix(yt, yp, labels=labels)
    
    plt.figure(figsize=(21,21))
    plot_confusion_matrix(cm, labels, normalize = True)
    plt.title(db_name)

#%%
base_strains_id = {x:i for i,x in enumerate(base_strains)}
for db_name, bn in feat_files.items():
    print(db_name)
    fname = os.path.join(save_dir, 'F_' + bn)
    feats = pd.read_csv(fname)

    col_feats = [x for x in feats.columns if x not in col2ignore]
    
    clf = RandomForestClassifier(n_estimators=1000)
    xx = feats[col_feats].values
    yy = feats['strain'].map(base_strains_id).values
    
    good = ~np.isnan(yy)
    
    xx = xx[good, :]
    yy = yy[good]
    scores = cross_val_score(clf, xx, yy, cv = cross_validation_fold)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    
#%%
classifiers_d = {}
for db_name, datasets  in sets2test.items():
    print(db_name, datasets['train'].shape)
    
    col_feats = [x for x in datasets['train'].columns if x not in col2ignore]
    
    X_train = datasets['train'][col_feats].values
    Y_train = datasets['train']['strain_id'].values
    
    x_test = datasets['test'][col_feats].values
    y_test = datasets['test']['strain_id'].values
    
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, Y_train)
    
    proba = clf.predict_proba(x_test)
    top_pred = np.argsort(proba, axis=1)[: ,::-1]
    preds = top_pred==y_test[:, np.newaxis]
    print(db_name, preds[:,0].mean())
    
    
    classifiers_d[db_name] = clf

#%%
for db_name, bn in feat_files.items():
    print(db_name)
    fname = os.path.join(save_dir, 'F_' + bn)
    feats = pd.read_csv(fname)

    col_feats = [x for x in feats.columns if x not in col2ignore]
    
    clf = RandomForestClassifier(n_estimators=1000)
    xx = feats[col_feats].values
    
    
    ss = np.sort(feats['strain'].unique())
    s_dict = {s:ii for ii,s in enumerate(ss)}
    feats['strain_id'] = feats['strain'].map(s_dict)
    yy = feats['strain_id'].values
    
    scores = cross_val_score(clf, xx, yy, cv = cross_validation_fold)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))