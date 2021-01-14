# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:31:55 2020

@author: ME
"""

# https://docs.dgl.ai/en/latest/tutorials/basics/5_hetero.html
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py
# https://docs.dgl.ai/en/0.4.x/tutorials/basics/5_hetero.html

import dgl
import numpy as np

# ratings = dgl.heterograph(
#     {('user', '+1', 'movie') : (np.array([0, 0, 1]), np.array([0, 1, 0])),
#      ('user', '-1', 'movie') : (np.array([2]), np.array([1]))})

import scipy.io
import urllib.request

data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = 'ACM.mat'

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
print(list(data.keys()))

print(type(data['PvsA']))
print('#Papers:', data['PvsA'].shape[0])
print('#Authors:', data['PvsA'].shape[1])
print('#Links:', data['PvsA'].nnz)


pa_g = dgl.heterograph({('paper', 'written-by', 'author') : data['PvsA'].nonzero()})

# data['PvsA'].nonzero() is a tupule of numpy arrays

# Nodes and edges are assigned integer IDs starting from zero and each type has its own counting.
# To distinguish the nodes and edges of different types, specify the type name as the argument.
print(pa_g.number_of_nodes('paper'))
# Canonical edge type name can be shortened to only one edge type name if it is
# uniquely distinguishable.
print(pa_g.number_of_edges(('paper', 'written-by', 'author')))
print(pa_g.number_of_edges('written-by'))
print(pa_g.successors(1, etype='written-by'))  # get the authors that write paper #1

# Type name argument could be omitted whenever the behavior is unambiguous.
print(pa_g.number_of_edges())  # Only one edge type, the edge type argument could be omitted


##### Paper-citing-paper graph is a homogeneous graph  #####
pp_g = dgl.heterograph({('paper', 'citing', 'paper') : data['PvsP'].nonzero()})
# equivalent (shorter) API for creating homogeneous graph
pp_g = dgl.from_scipy(data['PvsP'])

# All the ntype and etype arguments could be omitted because the behavior is unambiguous.
print(pp_g.number_of_nodes())
print(pp_g.number_of_edges())
print(pp_g.successors(3))


G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })

print(G)

# Question : How to add edge weights?

## SEMI SUpERVISED NODE CLASSIFICATION 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

pvc = data['PvsC'].tocsr()
# find all papers published in KDD, ICML, VLDB
c_selected = [0, 11, 13]  # KDD, ICML, VLDB
p_selected = pvc[:, c_selected].tocoo()
# generate labels
labels = pvc.indices
labels[labels == 11] = 1
labels[labels == 13] = 2
labels = torch.tensor(labels).long()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

import dgl.function as fn

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
    
    
class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict['paper']    
    
    
# Create the model. The output has three logits for three classes.
model = HeteroRGCN(G, 10, 10, 3)

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0
best_test_acc = 0

for epoch in range(100):
    logits = model(G)
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))