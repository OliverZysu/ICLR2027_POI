"""STAN model (adapted from official WWW'21 implementation, CPU-friendly)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HOURS = 24 * 7


class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max, max_len, device):
        super().__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max
        self.device = device

    def forward(self, self_attn, self_delta, traj_len):
        # self_delta: (N, M, L, emb) -> (N, L, M)
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)
        n, l_dim, m = self_delta.shape
        candidates = torch.arange(1, self.loc_max + 1, device=self.device).long()
        candidates = candidates.unsqueeze(0).expand(n, -1)
        emb_candidates = self.emb_loc(candidates)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)
        return self.value(attn).view(n, l_dim)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size):
        super().__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0 : traj_len[i], 0 : traj_len[i]] = 1
        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)
        attn = F.softmax(attn, dim=-1) * mask
        return torch.bmm(attn, self.value(joint))


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super().__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, mat2, vec, traj_len):
        delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0 : traj_len[i]] = 1
            delta_s[i, : traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i] - 1)[: traj_len[i]])

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vsu = (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtl = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtu = (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl + 1e-8)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl + 1e-8)
        return space_interval + time_interval


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super().__init__()
        (
            self.emb_t,
            self.emb_l,
            self.emb_u,
            self.emb_su,
            self.emb_sl,
            self.emb_tu,
            self.emb_tl,
        ) = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        traj = traj.clone()
        traj[:, :, 2] = (traj[:, :, 2] - 1) % HOURS + 1
        joint = self.emb_t(traj[:, :, 2]) + self.emb_l(traj[:, :, 1]) + self.emb_u(traj[:, :, 0])
        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0 : traj_len[i], 0 : traj_len[i]] = 1
        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vsu = (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtl = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtu = (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl + 1e-8)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl + 1e-8)
        return joint, space_interval + time_interval


class STAN(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, max_len, device):
        super().__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl
        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim - 1, embed_layers)
        self.Attn = Attn(emb_l, l_dim - 1, max_len, device)
        self.device = device

    def forward(self, traj, mat1, mat2, vec, traj_len):
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)
        self_attn = self.SelfAttn(joint, delta, traj_len)
        self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)
        return self.Attn(self_attn, self_delta, traj_len)
