
import torch
import torch.nn as nn
class TrajectoryEncoder(nn.Module):
    def __init__(self, n_poi, dim=128):
        super().__init__()
        self.poi_emb=nn.Embedding(n_poi, dim)
        self.gru=nn.GRU(dim, dim, batch_first=True)
    def forward(self,x):
        h,_=self.gru(self.poi_emb(x))
        return h[:,-1]
class POIPredictor(nn.Module):
    def __init__(self, n_poi, dim=128):
        super().__init__()
        self.out=nn.Linear(dim,n_poi)
    def forward(self,z): return self.out(z)
