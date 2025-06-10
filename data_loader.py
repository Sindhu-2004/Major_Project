import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import torch

def load_blogcatalog_data(filepath='data/BlogCatalog.mat'):
    """ Load BlogCatalog dataset from .mat file """
    data = sio.loadmat(filepath)
    adj = data['Network']  # Adjacency matrix (Graph)
    features = data['Attributes']  # Node features
    labels = data['Label'].flatten()  # Ground truth labels

    # Normalize adjacency matrix
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    return torch.FloatTensor(adj_norm.toarray()), torch.FloatTensor(features.toarray()), labels

def normalize_adj(adj):
    """ Normalize adjacency matrix """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
