""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Variational Graph Auto-Encoder
class VGAE(nn.Module):
   def __init__(self, **kwargs):
      super(VGAE, self).__init__()
      
      self.num_neurons = kwargs['num_neurons']
      self.num_features = kwargs['num_features']
      self.embedding_size = kwargs['embedding_size']
      
      self.w_0 = VGAE.random_uniform_init(self.num_features, self.num_neurons)
      self.b_0 = torch.nn.init.constant_(nn.Parameter(torch.Tensor(self.num_neurons)), 0.01)
      #mu mean
      #attention: 1. VAGE.r... 2. two #
      self.w_1_mu = VGAE.random_uniform_init(self.num_neurons, self.embedding_size)
      #B BIAS
      self.b_1_mu = torch.nn.init.constant_(nn.Parameter(torch.Tensor(self.embedding_size)), 0.01)
      #sigma standard deviation
      self.w_1_sigma = VGAE.random_uniform_init(self.num_neurons, self.embedding_size)
      self.b_1_sigma = torch.nn.init.constant_(nn.Parameter(torch.Tensor(self.embedding_size)), 0.01)

      
   @staticmethod
   def random_uniform_init(input_dim, output_dim):
      
      init_range = np.sqrt(6.0/(input_dim + output_dim))
      tensor = torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range)
      
      return nn.Parameter(tensor)

   def encode(self, adjacency, norm_adj, x_features):
      #H0=X, H1=AH, H2=AHW?
      hidden_0 = torch.relu(torch.add(torch.matmul(torch.matmul(norm_adj, x_features), self.w_0), self.b_0))
      self.GCN_mu = torch.add(torch.matmul(torch.matmul(norm_adj, hidden_0), self.w_1_mu), self.b_1_mu)
      #why?
      
      self.GCN_sigma = torch.exp(torch.add(torch.matmul(torch.matmul(norm_adj, hidden_0), self.w_1_sigma), self.b_1_sigma))
      
      z = self.GCN_mu + torch.randn(self.GCN_sigma.size()) * self.GCN_sigma
      
      return z
   
   @staticmethod
   def decode(z):
      x_hat = torch.sigmoid(torch.matmul(z, z.t()))
      return x_hat


   def forward(self, adjacency, norm_adj, x_features):
      z = self.encode(adjacency, norm_adj, x_features)
      x_hat = VGAE.decode(z)
      
      return x_hat


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(vcdn_feat)

        return output

    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
