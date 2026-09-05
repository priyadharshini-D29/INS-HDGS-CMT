"""
================================================================
INS-HDGS-CMT — Phase 8 Deep-Learning BASELINES
================================================================
Self-contained PyTorch reference implementations of standard EEG /
multimodal baselines, used to benchmark the full INS-HDGS-CMT model
under the *identical* LOSOCV folds, labels and metrics.

NO external deps beyond torch (braindecode / torcheeg NOT required).

EEG tensor convention (matches braindecode):
    x_eeg : (B, 1, C, T)   C = channels (19)   T = time samples (1500)

ET tensor convention:
    x_et  : (B, T_et, C_et) raw eye-tracking sequence (600, 6)

Each model is a binary classifier returning logits of shape (B, n_classes).

References
----------
EEGNet           Lawhern et al. 2018,  J. Neural Eng.
ShallowConvNet   Schirrmeister et al. 2017, Hum. Brain Mapp.
DeepConvNet      Schirrmeister et al. 2017, Hum. Brain Mapp.
CNN-BiLSTM       classic temporal hybrid (engagement/affect literature)
EarlyFusionMLP   concatenated-feature multimodal baseline
================================================================
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

class _Square(nn.Module):
    def forward(self, x):  # noqa: D401
        return x * x


class _Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


# ── EEGNet (Lawhern 2018) ──────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """Compact CNN baseline. Input (B, 1, C, T)."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 64, drop: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2),
                      bias=False),
            nn.BatchNorm2d(F1),
        )
        # depthwise spatial conv across channels
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        # separable conv
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        # infer flattened size
        with torch.no_grad():
            d = self.block3(self.block2(self.block1(
                torch.zeros(1, 1, n_chans, n_times))))
            self._flat = d.numel()
        self.classify = nn.Linear(self._flat, n_classes)

    def forward(self, x_eeg, x_et=None):
        x = self.block1(x_eeg)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classify(x)


# ── ShallowConvNet (Schirrmeister 2017) ─────────────────────────────────────────

class ShallowConvNet(nn.Module):
    """Strong oscillatory-feature baseline. Input (B, 1, C, T)."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 n_filters_time: int = 40, n_filters_spat: int = 40,
                 filter_time_length: int = 25, pool_time_length: int = 75,
                 pool_time_stride: int = 15, drop: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters_time, (1, filter_time_length))
        self.spatial = nn.Conv2d(n_filters_time, n_filters_spat,
                                 (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.square = _Square()
        self.pool = nn.AvgPool2d((1, pool_time_length), (1, pool_time_stride))
        self.log = _Log()
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_chans, n_times))
            self._flat = d.numel()
        self.classify = nn.Linear(self._flat, n_classes)

    def _features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        return self.drop(x)

    def forward(self, x_eeg, x_et=None):
        x = self._features(x_eeg).flatten(1)
        return self.classify(x)


# ── DeepConvNet (Schirrmeister 2017) ────────────────────────────────────────────

class DeepConvNet(nn.Module):
    """Deeper CNN baseline. Input (B, 1, C, T)."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 drop: float = 0.5):
        super().__init__()

        def conv_block(in_c, out_c, spat=False):
            layers = [nn.Conv2d(in_c, out_c,
                                (n_chans, 1) if spat else (1, 5),
                                bias=False)]
            layers += [nn.BatchNorm2d(out_c), nn.ELU(),
                       nn.MaxPool2d((1, 2)), nn.Dropout(drop)]
            return nn.Sequential(*layers)

        self.b1_time = nn.Conv2d(1, 25, (1, 5), bias=False)
        self.b1_spat = conv_block(25, 25, spat=True)
        self.b2 = conv_block(25, 50)
        self.b3 = conv_block(50, 100)
        self.b4 = conv_block(100, 200)
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_chans, n_times))
            self._flat = d.numel()
        self.classify = nn.Linear(self._flat, n_classes)

    def _features(self, x):
        x = self.b1_time(x)
        x = self.b1_spat(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return x

    def forward(self, x_eeg, x_et=None):
        x = self._features(x_eeg).flatten(1)
        return self.classify(x)


# ── CNN-BiLSTM hybrid ───────────────────────────────────────────────────────────

class CNNBiLSTM(nn.Module):
    """1-D CNN feature extractor → BiLSTM. Input (B, 1, C, T)."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 n_cnn: int = 32, hidden: int = 64, drop: float = 0.5):
        super().__init__()
        # collapse channels with a spatial conv, then temporal CNN
        self.spatial = nn.Conv2d(1, n_cnn, (n_chans, 1))
        self.temporal = nn.Sequential(
            nn.Conv1d(n_cnn, n_cnn, 7, padding=3), nn.BatchNorm1d(n_cnn),
            nn.ELU(), nn.MaxPool1d(4),
            nn.Conv1d(n_cnn, n_cnn, 7, padding=3), nn.BatchNorm1d(n_cnn),
            nn.ELU(), nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(n_cnn, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.classify = nn.Linear(hidden * 2, n_classes)

    def forward(self, x_eeg, x_et=None):
        x = self.spatial(x_eeg)            # (B, n_cnn, 1, T)
        x = x.squeeze(2)                   # (B, n_cnn, T)
        x = self.temporal(x)              # (B, n_cnn, T')
        x = x.transpose(1, 2)             # (B, T', n_cnn)
        out, _ = self.lstm(x)             # (B, T', 2H)
        x = out.mean(dim=1)               # temporal average pool
        return self.classify(self.drop(x))


# ── Early-fusion MLP (multimodal: EEG band-power + ET summary) ───────────────────

class EarlyFusionMLP(nn.Module):
    """
    Multimodal baseline. Consumes hand-crafted per-epoch features
    (EEG band-power stats + ET summary) concatenated into one vector.
    Input here is the already-flattened feature vector (B, F).
    """

    def __init__(self, in_dim: int, n_classes: int = 2,
                 hidden=(128, 64), drop: float = 0.5):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ELU(),
                       nn.Dropout(drop)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.classify = nn.Linear(d, n_classes)

    def forward(self, x_feat, x_et=None):
        return self.classify(self.mlp(x_feat))


# ── Brain-connectivity GCN (electrode nodes + Pearson-correlation edges) ─────────

class _DenseGCNLayer(nn.Module):
    """Spectral GCN layer on a dense, per-graph adjacency.
    H' = ELU( Â H W )  with Â = symmetric-normalised (A + I)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, H, A_hat):                     # H (B,C,Fin)  A_hat (B,C,C)
        return F.elu(torch.bmm(A_hat, self.lin(H)))  # (B,C,Fout)


class BrainGCN(nn.Module):
    """
    Graph-neural-network baseline matching the uploaded paper's design:
    19/24 electrodes as nodes, per-band log band-power as node features, and
    Pearson-correlation functional connectivity as (dense) edge weights.

    Input convention (kind="graph"): a packed tensor (B, C, F_node + C) where
        x[..., :F_node]  = node features        (B, C, F_node)
        x[..., F_node:]  = raw adjacency A      (B, C, C)
    The packing keeps the runner's single-tensor train/eval path unchanged.
    """

    def __init__(self, n_nodes: int, node_dim: int, n_classes: int = 2,
                 hidden: int = 64, drop: float = 0.5):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.gc1 = _DenseGCNLayer(node_dim, hidden)
        self.gc2 = _DenseGCNLayer(hidden, hidden)
        self.drop = nn.Dropout(drop)
        self.classify = nn.Linear(hidden, n_classes)

    @staticmethod
    def _normalize(A):
        """Symmetric normalisation of (A + I): D^-1/2 (A+I) D^-1/2."""
        B, C, _ = A.shape
        A = A + torch.eye(C, device=A.device).unsqueeze(0)
        deg = A.sum(-1).clamp(min=1e-6)               # (B,C)
        dinv = deg.pow(-0.5)
        return A * dinv.unsqueeze(1) * dinv.unsqueeze(2)

    def forward(self, x_packed, x_et=None):
        C, Fn = self.n_nodes, self.node_dim
        H = x_packed[..., :Fn]                        # (B,C,Fn)
        A = x_packed[..., Fn:Fn + C]                  # (B,C,C)
        A_hat = self._normalize(A)
        H = self.gc1(H, A_hat)
        H = self.drop(H)
        H = self.gc2(H, A_hat)
        H = H.mean(dim=1)                             # global mean pool over nodes
        return self.classify(self.drop(H))


# ══════════════════════════════════════════════════════════════════════════════
#  NAMED BASELINES requested by the Brain-Informatics study brief
#  (Pipeline 1: EEG encoders · Pipeline 2: ET encoders · Pipeline 3: fusion)
#  All are self-contained PyTorch references trained under the identical LOSOCV
#  folds / labels / metrics as the proposed INS-HDGS-CMT model.
# ══════════════════════════════════════════════════════════════════════════════


def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding (seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


# ── Pipeline 1: CNN-LSTM (uni-directional temporal hybrid) ──────────────────────

class CNNLSTM(nn.Module):
    """Spatial conv → temporal CNN → (uni-directional) LSTM. Input (B,1,C,T).

    The classic CNN-LSTM EEG decoder (distinct from the bidirectional
    CNN-BiLSTM already in the registry)."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 n_cnn: int = 32, hidden: int = 64, drop: float = 0.5):
        super().__init__()
        self.spatial = nn.Conv2d(1, n_cnn, (n_chans, 1))
        self.temporal = nn.Sequential(
            nn.Conv1d(n_cnn, n_cnn, 7, padding=3), nn.BatchNorm1d(n_cnn),
            nn.ELU(), nn.MaxPool1d(4),
            nn.Conv1d(n_cnn, n_cnn, 7, padding=3), nn.BatchNorm1d(n_cnn),
            nn.ELU(), nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(n_cnn, hidden, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(drop)
        self.classify = nn.Linear(hidden, n_classes)

    def _feat(self, x_eeg):
        x = self.spatial(x_eeg).squeeze(2)        # (B, n_cnn, T)
        x = self.temporal(x).transpose(1, 2)      # (B, T', n_cnn)
        out, _ = self.lstm(x)                      # (B, T', H)
        return out[:, -1]                          # last hidden state (B, H)

    def forward(self, x_eeg, x_et=None):
        return self.classify(self.drop(self._feat(x_eeg)))


# ── Pipeline 1: EEG Transformer ─────────────────────────────────────────────────

class EEGTransformer(nn.Module):
    """Spatial-conv tokeniser → Transformer encoder over time. Input (B,1,C,T)."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 d_model: int = 64, nhead: int = 4, depth: int = 2,
                 patch: int = 25, drop: float = 0.3):
        super().__init__()
        # collapse channels and tokenise time into non-overlapping patches
        self.spatial = nn.Conv2d(1, d_model, (n_chans, 1))          # (B,d,1,T)
        self.tokeniser = nn.Conv1d(d_model, d_model, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=drop, batch_first=True,
                                         activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_classes)
        self._d = d_model

    def _feat(self, x_eeg):
        x = self.spatial(x_eeg).squeeze(2)             # (B,d,T)
        x = self.tokeniser(x).transpose(1, 2)          # (B,L,d)
        B, L, d = x.shape
        x = x + _sinusoidal_pe(L, d).to(x.device).unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                 # prepend CLS token
        x = self.encoder(x)
        return self.norm(x[:, 0])                       # CLS representation

    def forward(self, x_eeg, x_et=None):
        return self.classify(self._feat(x_eeg))


# ── Pipeline 1: TSception (Ding et al. 2022, IEEE TAFFC) ────────────────────────

class TSception(nn.Module):
    """Multi-scale Temporal (Tception) + Spatial (Sception) convolutions.
    Input (B,1,C,T). Hemisphere split is approximated by a global + half-channel
    spatial kernel since the NeuMa montage order is not lateralised here."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 sampling_rate: int = 250, num_T: int = 15, num_S: int = 15,
                 hidden: int = 32, drop: float = 0.5):
        super().__init__()
        # three temporal scales (0.5, 0.25, 0.125 s)
        scales = [0.5, 0.25, 0.125]
        self.tception = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_T, (1, max(1, int(sampling_rate * s)))),
                nn.LeakyReLU(), nn.AvgPool2d((1, 8)))
            for s in scales])
        self.bn_t = nn.BatchNorm2d(num_T)
        # spatial: global (all channels) and hemisphere (half) kernels
        self.s_global = nn.Sequential(
            nn.Conv2d(num_T, num_S, (n_chans, 1)), nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        half = max(1, n_chans // 2)
        self.s_hemi = nn.Sequential(
            nn.Conv2d(num_T, num_S, (half, 1), stride=(half, 1)),
            nn.LeakyReLU(), nn.AvgPool2d((1, 2)))
        self.bn_s = nn.BatchNorm2d(num_S)
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            d = self._feat(torch.zeros(1, 1, n_chans, n_times))
            flat = d.numel()
        self.fc = nn.Sequential(nn.Linear(flat, hidden), nn.LeakyReLU(),
                                nn.Dropout(drop))
        self.classify = nn.Linear(hidden, n_classes)

    def _feat(self, x):
        # multi-scale temporal: concat along time after pooling to min length
        outs = [t(x) for t in self.tception]
        L = min(o.shape[-1] for o in outs)
        t = torch.cat([o[..., :L] for o in outs], dim=-1)   # (B,num_T,C,3L)
        t = self.bn_t(t)
        g = self.s_global(t)                                 # (B,num_S,1,*)
        h = self.s_hemi(t)                                   # (B,num_S,~2,*)
        z = torch.cat([g.flatten(1), h.flatten(1)], dim=1)
        return self.drop(z)

    def forward(self, x_eeg, x_et=None):
        z = self._feat(x_eeg)
        return self.classify(self.fc(z))


# ── Pipeline 1: GAT (Graph Attention Network over electrode nodes) ───────────────

class _GATLayer(nn.Module):
    """Multi-head graph attention on a dense, adjacency-masked graph.
    H (B,N,Fin) · mask (B,N,N) → (B,N,heads*out)."""

    def __init__(self, in_dim, out_dim, heads=4, drop=0.3, concat=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(1, heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(drop)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, H, mask):
        B, N, _ = H.shape
        h = self.W(H).view(B, N, self.heads, self.out_dim)     # (B,N,H,O)
        e_src = (h * self.a_src).sum(-1, keepdim=True)         # (B,N,H,1)
        e_dst = (h * self.a_dst).sum(-1, keepdim=True)
        # attention logits (B,H,N,N)
        e = self.leaky(e_src.permute(0, 2, 1, 3) + e_dst.permute(0, 2, 3, 1))
        m = (mask > 0).unsqueeze(1)                             # (B,1,N,N)
        e = e.masked_fill(~m, float("-inf"))
        att = self.drop(torch.softmax(e, dim=-1))              # (B,H,N,N)
        out = torch.einsum("bhij,bjho->biho", att, h)          # (B,N,H,O)
        if self.concat:
            return F.elu(out.reshape(B, N, self.heads * self.out_dim))
        return F.elu(out.mean(2))                              # (B,N,O)


class GAT(nn.Module):
    """Graph Attention Network baseline over electrode nodes.
    Packed input (B,C,F_node+C): node features + Pearson-corr adjacency
    (same packing convention as BrainGCN)."""

    def __init__(self, n_nodes: int, node_dim: int, n_classes: int = 2,
                 hidden: int = 16, heads: int = 4, drop: float = 0.3):
        super().__init__()
        self.n_nodes, self.node_dim = n_nodes, node_dim
        self.g1 = _GATLayer(node_dim, hidden, heads=heads, drop=drop, concat=True)
        self.g2 = _GATLayer(hidden * heads, hidden, heads=1, drop=drop, concat=False)
        self.drop = nn.Dropout(drop)
        self.classify = nn.Linear(hidden, n_classes)

    def forward(self, x_packed, x_et=None):
        C, Fn = self.n_nodes, self.node_dim
        H = x_packed[..., :Fn]                       # (B,C,Fn)
        A = x_packed[..., Fn:Fn + C]                 # (B,C,C)
        A = A + torch.eye(C, device=A.device).unsqueeze(0)   # self-loops
        H = self.g1(H, A)
        H = self.drop(H)
        H = self.g2(H, A)                            # (B,C,hidden)
        return self.classify(H.mean(1))              # global mean pool


# ── Pipeline 2: Eye-tracking sequence encoders.  Input (B, T_et, C_et) ──────────

class _ETRecurrent(nn.Module):
    """Shared scaffold for ET-LSTM / ET-GRU."""

    def __init__(self, in_dim: int, n_classes: int = 2, hidden: int = 64,
                 layers: int = 2, drop: float = 0.4, rnn: str = "lstm"):
        super().__init__()
        Rnn = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = Rnn(in_dim, hidden, num_layers=layers, batch_first=True,
                       bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.drop = nn.Dropout(drop)
        self.classify = nn.Linear(hidden * 2, n_classes)

    def _feat(self, x_et):
        out, _ = self.rnn(x_et)                       # (B,T,2H)
        return out.mean(1)                            # temporal mean pool

    def forward(self, x_et, x_unused=None):
        return self.classify(self.drop(self._feat(x_et)))


class ETLSTM(_ETRecurrent):
    def __init__(self, in_dim: int, n_classes: int = 2, **kw):
        super().__init__(in_dim, n_classes, rnn="lstm", **kw)


class ETGRU(_ETRecurrent):
    def __init__(self, in_dim: int, n_classes: int = 2, **kw):
        super().__init__(in_dim, n_classes, rnn="gru", **kw)


class ETTransformer(nn.Module):
    """Transformer encoder over the ET sequence — the ET branch used inside
    INS-HDGS-CMT, re-used here as a stand-alone baseline. Input (B,T_et,C_et)."""

    def __init__(self, in_dim: int, n_classes: int = 2, d_model: int = 64,
                 nhead: int = 4, depth: int = 2, drop: float = 0.3,
                 stride: int = 4):
        super().__init__()
        # downsample the long (600-step) ET sequence with a strided conv embed
        self.embed = nn.Conv1d(in_dim, d_model, kernel_size=stride, stride=stride)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=drop, batch_first=True,
                                         activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_classes)

    def _feat(self, x_et):
        x = self.embed(x_et.transpose(1, 2)).transpose(1, 2)   # (B,L,d)
        B, L, d = x.shape
        x = x + _sinusoidal_pe(L, d).to(x.device).unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        x = self.encoder(torch.cat([cls, x], dim=1))
        return self.norm(x[:, 0])

    def forward(self, x_et, x_unused=None):
        return self.classify(self._feat(x_et))


# ── Pipeline 3: multimodal fusion baselines.  forward(x_eeg, x_et) ──────────────

class LateFusion(nn.Module):
    """Decision-level late fusion: CNN-LSTM (EEG) + ET-LSTM (ET), logits averaged."""

    def __init__(self, n_chans: int, n_times: int, et_dim: int,
                 n_classes: int = 2):
        super().__init__()
        self.eeg = CNNLSTM(n_chans, n_times, n_classes)
        self.et = ETLSTM(et_dim, n_classes)

    def forward(self, x_eeg, x_et):
        return 0.5 * (self.eeg(x_eeg) + self.et(x_et))


class DualTransformerFusion(nn.Module):
    """Feature-level fusion of an EEG Transformer and an ET Transformer."""

    def __init__(self, n_chans: int, n_times: int, et_dim: int,
                 n_classes: int = 2, d_model: int = 64, drop: float = 0.3):
        super().__init__()
        self.eeg = EEGTransformer(n_chans, n_times, n_classes, d_model=d_model)
        self.et = ETTransformer(et_dim, n_classes, d_model=d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d_model, n_classes))

    def forward(self, x_eeg, x_et):
        fe = self.eeg._feat(x_eeg)
        ft = self.et._feat(x_et)
        return self.head(torch.cat([fe, ft], dim=-1))


class CrossAttentionFusion(nn.Module):
    """EEG tokens attend to ET tokens (and vice-versa) via cross-attention."""

    def __init__(self, n_chans: int, n_times: int, et_dim: int,
                 n_classes: int = 2, d_model: int = 64, nhead: int = 4,
                 patch: int = 25, et_stride: int = 4, drop: float = 0.3):
        super().__init__()
        self.eeg_spatial = nn.Conv2d(1, d_model, (n_chans, 1))
        self.eeg_tok = nn.Conv1d(d_model, d_model, patch, stride=patch)
        self.et_embed = nn.Conv1d(et_dim, d_model, et_stride, stride=et_stride)
        self.x_eeg2et = nn.MultiheadAttention(d_model, nhead, dropout=drop,
                                              batch_first=True)
        self.x_et2eeg = nn.MultiheadAttention(d_model, nhead, dropout=drop,
                                              batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d_model, n_classes))

    def forward(self, x_eeg, x_et):
        e = self.eeg_spatial(x_eeg).squeeze(2)           # (B,d,T)
        e = self.eeg_tok(e).transpose(1, 2)              # (B,Le,d)
        t = self.et_embed(x_et.transpose(1, 2)).transpose(1, 2)   # (B,Lt,d)
        e_att, _ = self.x_eeg2et(e, t, t)                # EEG queries ET
        t_att, _ = self.x_et2eeg(t, e, e)                # ET queries EEG
        z = torch.cat([self.norm(e_att).mean(1),
                       self.norm(t_att).mean(1)], dim=-1)
        return self.head(z)


class MultimodalTransformer(nn.Module):
    """Single Transformer over concatenated EEG+ET tokens with modality embeddings."""

    def __init__(self, n_chans: int, n_times: int, et_dim: int,
                 n_classes: int = 2, d_model: int = 64, nhead: int = 4,
                 depth: int = 2, patch: int = 25, et_stride: int = 4,
                 drop: float = 0.3):
        super().__init__()
        self.eeg_spatial = nn.Conv2d(1, d_model, (n_chans, 1))
        self.eeg_tok = nn.Conv1d(d_model, d_model, patch, stride=patch)
        self.et_embed = nn.Conv1d(et_dim, d_model, et_stride, stride=et_stride)
        self.mod_emb = nn.Parameter(torch.zeros(2, d_model))   # EEG / ET tags
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=drop, batch_first=True,
                                         activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_classes)

    def forward(self, x_eeg, x_et):
        e = self.eeg_spatial(x_eeg).squeeze(2)
        e = self.eeg_tok(e).transpose(1, 2) + self.mod_emb[0]
        t = self.et_embed(x_et.transpose(1, 2)).transpose(1, 2) + self.mod_emb[1]
        B = e.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, e, t], dim=1)
        x = self.encoder(x)
        return self.classify(self.norm(x[:, 0]))


class DynamicGATETFusion(nn.Module):
    """Proposed-style EEG branch (dynamic channel-graph attention learnt from raw
    EEG) fused with the ET Transformer. A lightweight stand-in for DynamicGAT+ET
    that consumes the same raw (B,1,C,T)+(B,T_et,C_et) inputs as the other
    fusion baselines (no pre-packed graph required)."""

    def __init__(self, n_chans: int, n_times: int, et_dim: int,
                 n_classes: int = 2, d_model: int = 64, nhead: int = 4,
                 drop: float = 0.3):
        super().__init__()
        # per-channel temporal embedding → node features
        self.chan_embed = nn.Sequential(
            nn.Conv1d(1, d_model, 25, stride=12), nn.ELU(),
            nn.AdaptiveAvgPool1d(1))                      # (B*C, d, 1)
        # dynamic graph = learned self-attention adjacency over channel nodes
        self.gat = nn.MultiheadAttention(d_model, nhead, dropout=drop,
                                         batch_first=True)
        self.eeg_norm = nn.LayerNorm(d_model)
        self.et = ETTransformer(et_dim, n_classes, d_model=d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d_model, n_classes))

    def forward(self, x_eeg, x_et):
        B, _, C, T = x_eeg.shape
        nodes = x_eeg.reshape(B * C, 1, T)
        nodes = self.chan_embed(nodes).squeeze(-1).reshape(B, C, -1)   # (B,C,d)
        att, _ = self.gat(nodes, nodes, nodes)                      # dynamic graph
        fe = self.eeg_norm(att).mean(1)                             # (B,d)
        ft = self.et._feat(x_et)                                    # (B,d)
        return self.head(torch.cat([fe, ft], dim=-1))


# ── registry ────────────────────────────────────────────────────────────────────

# kind: "eeg"        → forward(x_eeg) with x_eeg (B,1,C,T)
#       "feat"       → forward(x_feat) with precomputed feature vector (B,F)
#       "graph"      → forward(x_packed) with packed (B,C,F_node+C) feats+adjacency
#       "et"         → forward(x_et) with raw ET sequence (B,T_et,C_et)
#       "multimodal" → forward(x_eeg, x_et) with both raw modalities
# `pipeline` tags which study table the model belongs to (1=EEG, 2=ET, 3=fusion).
MODEL_REGISTRY = {
    # ── legacy / generic DL baselines (kept for back-compat) ──
    "eegnet":     dict(cls=EEGNet,         kind="eeg",   pipeline=1),
    "shallow":    dict(cls=ShallowConvNet, kind="eeg",   pipeline=1),
    "deep":       dict(cls=DeepConvNet,    kind="eeg",   pipeline=1),
    "cnn_bilstm": dict(cls=CNNBiLSTM,      kind="eeg",   pipeline=1),
    "fusion_mlp": dict(cls=EarlyFusionMLP, kind="feat",  pipeline=3),
    "brain_gcn":  dict(cls=BrainGCN,       kind="graph", pipeline=1),
    # ── Pipeline 1: EEG encoder comparison ──
    "cnn_lstm":        dict(cls=CNNLSTM,        kind="eeg",   pipeline=1),
    "eeg_transformer": dict(cls=EEGTransformer, kind="eeg",   pipeline=1),
    "tsception":       dict(cls=TSception,      kind="eeg",   pipeline=1),
    "gat":             dict(cls=GAT,            kind="graph", pipeline=1),
    # ── Pipeline 2: ET encoder comparison ──
    "et_lstm":        dict(cls=ETLSTM,        kind="et", pipeline=2),
    "et_gru":         dict(cls=ETGRU,         kind="et", pipeline=2),
    "et_transformer": dict(cls=ETTransformer, kind="et", pipeline=2),
    # ── Pipeline 3: multimodal fusion comparison ──
    "late_fusion":      dict(cls=LateFusion,            kind="multimodal", pipeline=3),
    "dual_transformer": dict(cls=DualTransformerFusion, kind="multimodal", pipeline=3),
    "cross_attention":  dict(cls=CrossAttentionFusion,  kind="multimodal", pipeline=3),
    "mm_transformer":   dict(cls=MultimodalTransformer, kind="multimodal", pipeline=3),
    "dynamicgat_et":    dict(cls=DynamicGATETFusion,    kind="multimodal", pipeline=3),
}

# Convenience groups for the runners / launchers.
PIPELINE_MODELS = {
    1: ["cnn_lstm", "eeg_transformer", "tsception", "gat"],
    2: ["et_lstm", "et_gru", "et_transformer"],
    3: ["late_fusion", "dual_transformer", "cross_attention",
        "mm_transformer", "dynamicgat_et"],
}
