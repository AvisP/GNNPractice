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

ratings = dgl.heterograph(
    {('user', '+1', 'movie') : (np.array([0, 0, 1]), np.array([0, 1, 0])),
     ('user', '-1', 'movie') : (np.array([2]), np.array([1]))})

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