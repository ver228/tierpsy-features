#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018

@author: ajaver

Modified from: 
    https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/2_logistic_regression.py
"""


import numpy as np

import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit

def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]




import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from compare_ftests import col2ignore

if __name__ == "__main__":
    
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../data/SWDB'
    feat_files = {
            'OW' : 'F0.025_ow_features_old_SWDB.csv',
            'tierpsy' :'F0.025_tierpsy_features_SWDB.csv'
            }
    
    
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        #maybe i should divided it in train and test, but cross validation should be enough...
        feats['set_type'] = ''
        
        feat_data[db_name] = feats
        
    col2ignore_r = col2ignore + ['strain_id', 'set_type']
    #%% scale data
    for db_name, feats in feat_data.items(): 
        col_val = [x for x in feats.columns if x not in col2ignore_r]
        
        dd = feats[col_val]
        z = (dd-dd.mean())/(dd.std())
        feats[col_val] = z
        feat_data[db_name] = feats

    #%% create a dataset with all the features
    feats = feat_data['OW']
    col_feats = [x for x in feats.columns if x not in col2ignore_r]
    feats = feats[col_feats + ['base_name']]
    feat_data['all'] = feat_data['tierpsy'].merge(feats, on='base_name')
    
    #%%
    n_estimators = 1000
    n_jobs = 12
    n_splits = 10
    batch_size = 250
    n_epochs = 500
    
    results = {}
    for db_name, feats in feat_data.items():
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        n_features = X.shape[1]
        n_classes = y.shape[0]
        
        res = []
        sss = StratifiedShuffleSplit(n_splits = n_splits, test_size = 0.2, random_state=777)
        for ii, (train_index, test_index) in enumerate(sss.split(X, y)):
            if db_name != 'tierpsy':
                continue
            print(db_name, ii + 1, n_splits)
            #%%
            x_train, y_train  = X[train_index], y[train_index]
            x_test, y_test  = X[test_index], y[test_index]
            
            
            n_examples = train_index.size
            
            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).long()
            
            #this data I can leave it in the gpu()
            x_test = torch.from_numpy(x_test).float().cuda() 
            y_test = torch.from_numpy(y_test).long().cuda() 
            
            
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            model = build_model(n_features, n_classes)
            loss = torch.nn.CrossEntropyLoss(size_average=True)
            
            model = model.cuda()
            loss = loss.cuda()
            
            optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
            #optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            
            
            for i in range(n_epochs):
                cost = 0.
                num_batches = n_examples // batch_size
                for k, (xx, yy) in enumerate(loader):
                    xx = xx.cuda()
                    yy = yy.cuda()
                    
                    start, end = k * batch_size, (k + 1) * batch_size
                    cost += train(model, loss, optimizer, xx, yy)
                
                
                
                if i % 100 == 99 or i == n_epochs-1:
                    dd = Variable(x_test, requires_grad=False)
                    output = model.forward(dd)
                    y_pred_proba = output.data.cpu().numpy()
                    y_pred = y_pred_proba.argmax(axis=1)
                    
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    acc = 100. * np.mean(y_pred == y_test)
                    print("Epoch %d, cost = %f, acc = %.2f%%, f1 = %.2f"
                          % (i + 1, cost / num_batches, acc, f1))
            
            res.append((y_test, y_pred_proba, model))
        
        results[db_name] = (res, col_feats)
            
            