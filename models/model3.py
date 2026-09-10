
"""Pattern graph disentanglement: shared transition + personal residual."""
import torch, torch.nn as nn
from .common import TrajectoryEncoder, POIPredictor
class Model3(nn.Module):
    def __init__(self,n_poi,dim=128):
        super().__init__()
        self.encoder=TrajectoryEncoder(n_poi,dim)
        self.shared=nn.Sequential(nn.Linear(dim,dim),nn.ReLU())
        self.personal=nn.Linear(dim,dim)
        self.pred=POIPredictor(n_poi,dim)
    def forward(self,x):
        t=self.encoder(x)
        return self.pred(self.shared(t)+self.personal(t))
