
"""Shared mobility space with user-specific projection."""
import torch, torch.nn as nn
from .common import TrajectoryEncoder, POIPredictor
class Model2(nn.Module):
    def __init__(self,n_poi,dim=128):
        super().__init__()
        self.encoder=TrajectoryEncoder(n_poi,dim)
        self.shared=nn.Parameter(torch.randn(dim,dim))
        self.proj=nn.Linear(dim,dim*dim)
        self.pred=POIPredictor(n_poi,dim)
    def forward(self,x):
        t=self.encoder(x); P=self.proj(t).view(-1,t.size(-1),t.size(-1))
        z=torch.bmm(P,self.shared.expand(t.size(0),-1,-1).mean(-1).unsqueeze(-1)).squeeze(-1)+t
        return self.pred(z)
