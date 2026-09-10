"""GETNext model components (adapted from official PyTorch implementation)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))
        A = A + 1
        e = e * A
        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        return output + self.bias if self.bias is not None else output


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super().__init__()
        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        channels = [ninput] + list(nhid) + [noutput]
        for i in range(len(channels) - 1):
            self.gcn.append(GraphConvolution(channels[i], channels[i + 1]))

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)
        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_idx):
        return self.user_embedding(user_idx)


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super().__init__()
        self.cat_embedding = nn.Embedding(num_cats, embedding_dim)

    def forward(self, cat_idx):
        return self.cat_embedding(cat_idx)


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super().__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        return self.leaky_relu(x)


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, None, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super().__init__()
        assert activation == "sin"
        self.l1 = SineActivation(1, out_dim)

    def forward(self, x):
        return self.l1(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        return self.decoder_poi(x), self.decoder_time(x), self.decoder_cat(x)


class GETNext(nn.Module):
    """Wrapper holding all GETNext submodules for convenient save/load."""

    def __init__(
        self,
        gcn_nfeat,
        gcn_nhid,
        poi_embed_dim,
        gcn_dropout,
        node_attn_nhid,
        num_users,
        user_embed_dim,
        time_embed_dim,
        num_cats,
        cat_embed_dim,
        num_pois,
        transformer_nhead,
        transformer_nhid,
        transformer_nlayers,
        transformer_dropout,
    ):
        super().__init__()
        self.poi_embed_model = GCN(gcn_nfeat, gcn_nhid, poi_embed_dim, gcn_dropout)
        self.node_attn_model = NodeAttnMap(gcn_nfeat, node_attn_nhid, use_mask=False)
        self.user_embed_model = UserEmbeddings(num_users, user_embed_dim)
        self.time_embed_model = Time2Vec("sin", out_dim=time_embed_dim)
        self.cat_embed_model = CategoryEmbeddings(num_cats, cat_embed_dim)
        self.embed_fuse_model1 = FuseEmbeddings(user_embed_dim, poi_embed_dim)
        self.embed_fuse_model2 = FuseEmbeddings(time_embed_dim, cat_embed_dim)
        seq_dim = poi_embed_dim + user_embed_dim + time_embed_dim + cat_embed_dim
        self.seq_model = TransformerModel(
            num_pois,
            num_cats,
            seq_dim,
            transformer_nhead,
            transformer_nhid,
            transformer_nlayers,
            dropout=transformer_dropout,
        )
