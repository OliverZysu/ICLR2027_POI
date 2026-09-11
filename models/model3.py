"""Pattern graph disentanglement: shared transition + personal residual."""
import torch.nn as nn

from .common import TrajectoryEncoder, POIPredictor


class Model3(nn.Module):
    def __init__(self, n_poi, dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(n_poi, dim)
        self.shared = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.personal = nn.Linear(dim, dim)
        self.pred = POIPredictor(n_poi, dim)

    def forward(self, x, lengths=None, region=None):
        del region  # unused
        t = self.encoder(x, lengths)
        return self.pred(self.shared(t) + self.personal(t))
