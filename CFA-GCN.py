import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import time
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from genera_data import  *
import matplotlib
matplotlib.rc("font",family='WenQuanYi Micro Hei')
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from torch.autograd import Variable


path ='/home/lcf/Desktop/BSPC-get/'
dataset = 'Ewords'
# labels_name =['你','去','天','头','来','水','说']
labels_name =['apple', 'book', 'come', 'cup', 'go', 'head', 'stand', 'water', 'you']
n_splits = 5
import sys
import torch

n_features =500
# hidden_dim = 4
dropout = 0.05
num_class = 9
epochs = 200
early_stop = 10
weight_decay = 5e-4
learning_rate = 0.0001
batch_size = 80


x, y, adjs = load_adj(path,dataset)
print('tx, ty, ta shape',x.shape,y.shape,adjs.shape)
tx, ty,ta = load_test_adj(path,dataset)
train_ids = TensorDataset(x, y,adjs)
train_loader = DataLoader(dataset=train_ids, batch_size = batch_size, shuffle=True)
num_nodes = x.shape[2]
# batch_size = len(tx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    def __init__(self,in_features, num_nodes,out_features,bias: float = 0.0):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor( in_features * self.num_nodes, self.num_nodes * out_features)
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor( self.num_nodes * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    def forward(self , inputs, adj):
        bat_long = inputs.size(0)

        inputs = torch.reshape(inputs,(-1,self.num_nodes ,self.in_features))

        support = torch.bmm(adj, inputs)
        support = torch.reshape(support,(bat_long,self.num_nodes * self.in_features))

        output = torch.mm(support, self.weight)
        outputs = torch.reshape(output, (bat_long,self.num_nodes,self.out_features))

        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

class CFGCN(nn.Module):
    def __init__(self, seq_len,num_nodes,hidden_size,out_size,num_class):
        super(CFGCN, self).__init__()
        self.hidden = hidden_size
        self.out_size = out_size
        self._num_nodes = num_nodes
        self.gcn1 = GraphConvolution(seq_len, num_nodes,hidden_size)
        self.gcn2 = GraphConvolution(hidden_size,num_nodes,out_size)
        self.gcn3 = GraphConvolution(out_size, num_nodes, 132)
        self.fc = nn.Linear(64 * 132,num_class)
        self.dropout = nn.Dropout(p = 0.05)

        pass

    def forward(self, X, adj):
        X1 = F.relu(self.gcn1(X, adj))
        X2 = F.tanh( self.gcn2(X1, adj))
        X3 = F.relu(self.gcn3 (X2, adj))

        x2 = X2.transpose(2, 1)
        s1 = torch.bmm(x2,X3)

        s = torch.reshape(s1,(-1,64 * 132))
        out = self.fc(s)
        return out
        # return F.log_softmax(X, dim = 1)

model  = CFGCN(1500,16,324,64,num_class)

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay = weight_decay)

kf = KFold(n_splits = n_splits)
torch.cuda.empty_cache()
for i in range(epochs):
    for train_index, data in enumerate(train_loader, 1):
        t_sum = 0
        v_sum = 0
        x_data, x_label, x_adj = data
        for train_index,valid_index in kf.split(x_data):
            train_data = x_data[train_index]  #y_test is train label
            train_labels1 = x_label[train_index]
            train_adj1 = x_adj[train_index]

            val_data = x_data[valid_index]
            val_labels1 = x_label[valid_index]
            val_adj1 = x_adj[valid_index]
            # print('train:%s,valid:%s'%(train_index,valid_index))
            train_acc = 0
            train_total = 0

            output = model(train_data.data, train_adj1)
            loss = criterion(output, train_labels1)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()

            q, pre = torch.max(output, 1)
            train_total += train_labels1.size(0)
            train_acc += (pre == train_labels1).sum().item()
            s1 = train_acc / train_total
            t_sum = t_sum + s1

            model.eval()
            v_correct = 0
            v_total = 0
            v_labels2 = val_labels1
            val_data = val_data
            v_outputs = model(val_data.data, val_adj1)
            v_, v_predicted = torch.max(v_outputs, 1)
            v_total += v_labels2.size(0)
            v_correct += (v_predicted == v_labels2).sum().item()
            s2 = v_correct / v_total
            v_sum = v_sum + s2
            # print('train acc', s1, 'val acc',s2)

        print(i,'train acc',t_sum/n_splits,'val acc', v_sum/n_splits )
        with torch.no_grad():
            correct = 0
            total = 0
            labels = ty
            outputs = model(tx.data, ta)  # ,tadjs
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('test acc----------------------------', correct / total)
            # pred_y = predicted
    # cm = confusion_matrix(labels, pred_y )
    # # print(cm)
    # plot_confusion_matrix(cm,target_names =labels_name,m=i)
