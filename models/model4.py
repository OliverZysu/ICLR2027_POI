
"""Business/spatio cluster aware projection model."""
import torch, torch.nn as nn
from .common import TrajectoryEncoder, POIPredictor
class Model4(nn.Module):
    def __init__(self,n_poi,n_region=64,dim=128):
        super().__init__()
        self.encoder=TrajectoryEncoder(n_poi,dim)
        self.region=nn.Embedding(n_region,dim)
        self.pred=POIPredictor(n_poi,dim)
    def forward(self,x,region=None):
        t=self.encoder(x)
        if region is None: r=torch.zeros_like(t)
        else: r=self.region(region)
        return self.pred(t+r)
