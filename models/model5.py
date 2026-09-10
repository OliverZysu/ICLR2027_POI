
"""Adaptive additive vs projection mixture of shared/specific preference."""
import torch, torch.nn as nn
from .common import TrajectoryEncoder, POIPredictor
class Model5(nn.Module):
    def __init__(self,n_poi,dim=128):
        super().__init__()
        self.encoder=TrajectoryEncoder(n_poi,dim)
        self.shared=nn.Parameter(torch.randn(dim))
        self.specific=nn.Linear(dim,dim)
        self.project=nn.Linear(dim,dim)
        self.gate=nn.Linear(dim,1)
        self.pred=POIPredictor(n_poi,dim)
    def forward(self,x):
        t=self.encoder(x)
        add=t+self.shared
        proj=self.project(t*self.shared)
        g=torch.sigmoid(self.gate(t))
        return self.pred(g*add+(1-g)*proj)
