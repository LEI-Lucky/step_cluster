import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(AutoEncoder, self).__init__()
        
        # 编码器层
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 潜在层
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器层
        decoder_layers = []
        hidden_dims.reverse()
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x_recon.size(0), 1, 28, 28)  # MNIST 数据集的形状 / 另外的数据集需要根据情况设置
        return x_recon, z

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, n_features, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, n_features))  # 聚类中心
        nn.init.xavier_normal_(self.clusters)   # 聚类中心的初始化，正态分布
        
    def forward(self, x):
        # 计算样本与聚类中心的距离
        q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.clusters, 2), 2))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

class DeepClusteringModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super(DeepClusteringModel, self).__init__()
        self.autoencoder = AutoEncoder(input_dim, hidden_dims, latent_dim)
        self.clustering = ClusteringLayer(n_clusters, latent_dim)
        
    def forward(self, x):
        x_recon, z = self.autoencoder(x)
        q = self.clustering(z)
        return x_recon, z, q

def save_model(model, path):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_model(model, path):
    """加载模型"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model 