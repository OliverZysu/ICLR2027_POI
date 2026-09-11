"""Pattern Cluster Projection Network.
Learns user trajectory prototypes and separates shared pattern from user residual.
"""
import torch
import torch.nn as nn

from .common import TrajectoryEncoder, POIPredictor


class Model1(nn.Module):
    def __init__(self, n_poi, n_pattern=32, dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(n_poi, dim)
        self.pattern = nn.Parameter(torch.randn(n_pattern, dim))
        self.assign = nn.Linear(dim, n_pattern)
        self.user = nn.Linear(dim, dim)
        self.pred = POIPredictor(n_poi, dim)

    def forward(self, x, lengths=None, region=None):
        del region  # unused
        t = self.encoder(x, lengths)
        a = torch.softmax(self.assign(t), -1)
        shared = a @ self.pattern
        z = shared + self.user(t)
        return self.pred(z)
