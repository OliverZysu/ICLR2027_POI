import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TrajectoryEncoder(nn.Module):
    def __init__(self, n_poi, dim=128):
        super().__init__()
        # index ``n_poi`` is reserved for padding
        self.pad_idx = n_poi
        self.poi_emb = nn.Embedding(n_poi + 1, dim, padding_idx=n_poi)
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x, lengths=None):
        emb = self.poi_emb(x)
        if lengths is None:
            h, _ = self.gru(emb)
            return h[:, -1]
        # lengths: (B,) real history length; x is left-padded so last token is real
        lengths = lengths.clamp(min=1)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return h_n[-1]


class POIPredictor(nn.Module):
    def __init__(self, n_poi, dim=128):
        super().__init__()
        self.out = nn.Linear(dim, n_poi)

    def forward(self, z):
        return self.out(z)
