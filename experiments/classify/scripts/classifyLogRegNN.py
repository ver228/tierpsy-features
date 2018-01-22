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
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from compare_ftests import col2ignore
from sklearn.model_selection import StratifiedShuffleSplit

import tqdm

class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



def _train_step(model, criterion, optimizer, input_v, target_v, cuda_id = 0):
    
    target_v = target_v.cuda(cuda_id)
    input_v = input_v.cuda(cuda_id)
    
    input_v = Variable(input_v, requires_grad=False)
    target_v = Variable(target_v, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    output = model(input_v)
    loss = criterion(output, target_v)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.data[0]

def fit_model(model, criterion, optimizer, loader, cuda_id = 0):
    pbar = tqdm.trange(n_epochs)
    for i in pbar:
        #Train model
        model.train()
        train_loss = 0.
        for k, (xx, yy) in enumerate(loader):
            train_loss += _train_step(model, criterion, optimizer, xx, yy, cuda_id=cuda_id)
        train_loss /= len(loader)
        
        d_str = "train loss = %f" % (train_loss)
        pbar.set_description(d_str)
    
    return model

def eval_model(model, criterion, input_v, target_v):
    #Eval model
    model.eval()
    output = model(input_v)
    
    loss = criterion(output, target_v).data[0]
    
    _, y_pred = output.max(dim=1)
    acc = (y_pred == target_v).float().mean().data[0]*100
    
    y_test_l, y_pred_l = target_v.cpu().data.numpy(), y_pred.cpu().data.numpy()
    f1 = f1_score(y_test_l, y_pred_l, average='weighted')
    
    return loss, acc, f1

def get_feat_importance(model, criterion, input_v, target_v):
    n_features = model.fc.in_features
    n_classes = model.fc.out_features
    
    model_reduced = SimpleNet(n_features-1, n_classes)
    model_reduced.eval()
    
    inds = list(range(n_features))
    res_selection = []
    for ii in range(n_features):
        ind_r = inds[:ii] + inds[ii+1:]
        model_reduced.fc.weight.data = model.fc.weight[:, ind_r].data
        input_r = input_v[:, ind_r]
        
        loss, acc, f1 = eval_model(model_reduced, criterion, input_r, target_v)
        res_selection.append((loss, acc, f1))
    
    loss, acc, f1 = map(np.array, zip(*res_selection))
    
    
    return dict(loss = loss, acc = acc, f1 = f1)

def remove_feats(importance_metrics, metric2exclude, input_v, input_train, col_feats_o):
    metrics = importance_metrics[metric2exclude]
    
    #remove the least important feature
    if metric2exclude == 'loss':
        ind2exclude = np.argmin(metrics)
    else:
        ind2exclude = np.argmax(metrics)
    
    
    
    inds = list(range(n_features)) 
    ind_r = inds[:ind2exclude] + inds[ind2exclude+1:]
    input_r = input_v[:, ind_r]
    input_train_r = input_train[:, ind_r]
    
    col_feats_r = col_feats_o.copy()
    feat2exclude = col_feats_r.pop(ind2exclude)
    
    return input_r, input_train_r, col_feats_r, feat2exclude

if __name__ == "__main__":
    
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/SWDB/'
    save_dir = '../data/SWDB'
    feat_files = {
            'OW' : 'F0.025_ow_features_old_SWDB.csv',
            'tierpsy' :'F0.025_tierpsy_features_SWDB.csv'
            }
    #%%
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        #if db_name == 'tierpsy':
        #    col_curv = [x for x in feats if 'path_curvature' in x]
        #    feats[col_curv] = np.log10(np.abs(feats[col_curv]) + 1e-5)
        
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
    n_folds = 1
    batch_size = 250
    n_epochs = 50
    
    cuda_id = 1
    
    metric2exclude = 'loss'
    #metric2exclude = 'acc'
    
    criterion = F.nll_loss
    
    results = {}
    for db_name, feats in feat_data.items():
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        y = feats['strain_id'].values
        X = feats[col_feats].values
        
        n_classes = int(y.max() + 1)
        
        cross_v_res = []
        sss = StratifiedShuffleSplit(n_splits = n_folds, test_size = 0.2, random_state=777)
        for i_fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            print(db_name, i_fold + 1, n_folds)
            
            x_train, y_train  = X[train_index], y[train_index]
            x_test, y_test  = X[test_index], y[test_index]
            
            
            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).long()
            
            x_test = torch.from_numpy(x_test).float()
            y_test = torch.from_numpy(y_test).long() 
            
            input_v = Variable(x_test.cuda(cuda_id), requires_grad=False)
            target_v = Variable(y_test.cuda(cuda_id), requires_grad=False)
            
            input_train = x_train
            target_train = y_train
            
            col_feats_r = col_feats
            
            
            fold_res = []
            while col_feats_r: #continue as long as there are any feature to remove
                n_features = input_v.size(1)
                
                model = SimpleNet(n_features, n_classes)
                model = model.cuda(cuda_id)
                
                optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
                
                dataset = TensorDataset(input_train, target_train)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                model = fit_model(model, criterion, optimizer, loader, cuda_id = cuda_id)
                res = eval_model(model, criterion, input_v, target_v)
                
                print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*res))
                print(metric2exclude, i_fold + 1, n_features)
                
                
                if n_features > 1:
                    importance_metrics = get_feat_importance(model, criterion, input_v, target_v)
                    
                    input_v, input_train, col_feats_r, feat2remove = \
                    remove_feats(importance_metrics, metric2exclude, 
                                 input_v, 
                                 input_train, 
                                 col_feats_r)
                
                
                    fold_res.append((feat2remove, res))
                else:
                    fold_res.append((col_feats_r[0], res))
                    
            cross_v_res.append(fold_res)
        results[db_name] = cross_v_res
        
    #%%