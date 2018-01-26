#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018S

@author: ajaver
S
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

from sklearn.metrics import f1_score

import tqdm

class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class TrainerSimpleNet():
    def __init__(self, n_classes, n_features, n_epochs = 250, batch_size = 250, cuda_id = 0):
        
        self.model = SimpleNet(n_features, n_classes)
        self.model = self.model.cuda(cuda_id)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9)
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cuda_id = cuda_id
        
        self.criterion = F.nll_loss
        
    def fit(self, input_train, target_train):
        dataset = TensorDataset(input_train, target_train)
        loader = DataLoader(dataset, batch_size = self.batch_size, shuffle=True)
        
        pbar = tqdm.trange(self.n_epochs)
        for i in pbar:
            #Train model
            self.model.train()
            train_loss = 0.
            for k, (xx, yy) in enumerate(loader):
                train_loss += self._train_step(xx, yy)
            train_loss /= len(loader)
            
            d_str = "train loss = %f" % (train_loss)
            pbar.set_description(d_str)
    
    def _train_step(self, input_v, target_v):
        target_v = target_v.cuda(self.cuda_id)
        input_v = input_v.cuda(self.cuda_id)
        
        input_v = Variable(input_v, requires_grad=False)
        target_v = Variable(target_v, requires_grad=False)
    
        # Reset gradient
        self.optimizer.zero_grad()
    
        # Forward
        output = self.model(input_v)
        loss = self.criterion(output, target_v)
    
        # Backward
        loss.backward()
    
        # Update parameters
        self.optimizer.step()
    
        return loss.data[0]
    
    def evaluate(self, input_v, target_v):
        return self._evaluate(self.model, input_v, target_v)
        
    def _evaluate(self, model, input_v, target_v):
        model.eval()
        output = model(input_v)
        
        loss = self.criterion(output, target_v).data[0]
        
        _, y_pred = output.max(dim=1)
        acc = (y_pred == target_v).float().mean().data[0]*100
        
        y_test_l, y_pred_l = target_v.cpu().data.numpy(), y_pred.cpu().data.numpy()
        f1 = f1_score(y_test_l, y_pred_l, average='weighted')
        
        return loss, acc, f1

    def get_feat_importance(self, input_v, target_v):
        n_features = self.model.fc.in_features
        n_classes = self.model.fc.out_features
        
        model_reduced = SimpleNet(n_features-1, n_classes)
        model_reduced.eval()
        
        inds = list(range(n_features))
        res_selection = []
        for ii in range(n_features):
            ind_r = inds[:ii] + inds[ii+1:]
            model_reduced.fc.weight.data = self.model.fc.weight[:, ind_r].data
            input_r = input_v[:, ind_r]
            
            loss, acc, f1 = self._evaluate(model_reduced, input_r, target_v)
            res_selection.append((loss, acc, f1))
        
        loss, acc, f1 = map(np.array, zip(*res_selection))
    
    
        return dict(loss = loss, acc = acc, f1 = f1)

def remove_feats(importance_metrics, 
                 metric2exclude, 
                 input_v, 
                 input_train, 
                 col_feats_o,
                 n_feats2remove):
    metrics = importance_metrics[metric2exclude]
    
    ind_orders = np.argsort(metrics)
    
    #remove the least important feature
    if metric2exclude != 'loss':
        ind_orders = ind_orders[::-1]
    
    ind2exclude = ind_orders[:n_feats2remove].tolist()
    
    n_features = input_v.size(1)
    ind_r = [x for x in range(n_features) if x not in ind2exclude]
    
    input_r = input_v[:, ind_r]
    input_train_r = input_train[:, ind_r]
    
    
    col_feats_r, feat2exclude = [], []
    for ii, f in enumerate(col_feats_o):
        if ii in ind2exclude:
            feat2exclude.append(f)
        else:
            col_feats_r.append(f)
    
    return input_r, input_train_r, col_feats_r, feat2exclude

def softmax_RFE(data_in):
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), col_feats_r = fold_data
    (cuda_id, n_epochs, batch_size, metric2exclude, n_feats2remove) = fold_param
    
    
    n_classes = int(y_train.max() + 1)
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long() 
    
    input_v = Variable(x_test.cuda(cuda_id), requires_grad=False)
    target_v = Variable(y_test.cuda(cuda_id), requires_grad=False)
    
    input_train = x_train
    target_train = y_train
    
    
    
    fold_res = []
    while len(col_feats_r)>1: #continue as long as there are any feature to remove
        
        n_features = input_v.size(1)
        trainer = TrainerSimpleNet(n_classes, n_features, n_epochs, batch_size, cuda_id)
        trainer.fit(input_train, target_train)
        res = trainer.evaluate(input_v, target_v)
        
        print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*res))
        print(metric2exclude, i_fold + 1, n_features)
        
        
        if n_features > 1:
            if n_feats2remove == 'log2':
                n2remove =  n_features - int(2**np.floor(np.log2(n_features - 1e-5))) #lowest power of 2
            else:
                n2remove = n_feats2remove
            
            
            importance_metrics = trainer.get_feat_importance(input_v, target_v)
            
            input_v, input_train, col_feats_r, feat2remove = \
            remove_feats(importance_metrics, 
                         metric2exclude, 
                         input_v, 
                         input_train, 
                         col_feats_r, 
                         n_feats2remove = n2remove)
        
        
            fold_res.append((feat2remove, res))
        else:
            fold_res.append((col_feats_r[0], res))
    
    return fold_id, fold_res
#%%
    

