"""1D-token autoencoder: the VA-VAE convolutional encoder and decoder, unchanged,
around a Perceiver-style bottleneck.

    conv encoder -> (B, C_enc, H, W) -> H*W tokens
      -> K learned latents cross-attend to the tokens        (Perceiver resampler)
      -> z = (B, embed_dim, K, 1)                             the 1D latent
      -> H*W learned position queries cross-attend to z      (Perceiver-IO decoding)
      -> (B, C_enc, H, W) -> conv decoder -> image

The latent is a KL-regularised Gaussian (per-token mean and log-variance), as in
the 2D arms; losses, distillation and training are inherited from AutoencoderKL.
"""

import torch
import torch.nn as nn

from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention + MLP. Queries `q` attend to key/values `kv`."""
    def __init__(self, dim, heads, kv_dim=None, mlp_ratio=4.0):
        super().__init__()
        kv_dim = kv_dim or dim
        self.ln_q  = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.attn  = nn.MultiheadAttention(dim, heads, kdim=kv_dim, vdim=kv_dim,
                                           batch_first=True)
        self.ln2   = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                   nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, q, kv):
        kv_n = self.ln_kv(kv)
        q = q + self.attn(self.ln_q(q), kv_n, kv_n, need_weights=False)[0]
        q = q + self.mlp(self.ln2(q))
        return q


class SelfAttnBlock(nn.Module):
    """Pre-LN self-attention + MLP over a token set."""
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2  = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                  nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        xn = self.ln1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class ConvTokenPoolAutoencoderKL(AutoencoderKL):
    """VA-VAE conv encoder/decoder + attention-pool bottleneck → K 1D tokens.

    Extra params vs AutoencoderKL:
      num_tokens : K, the number of latent tokens.
      attn_heads : heads for the pool/unpool attention.
      attn_depth : number of (cross-attn, self-attn) layers on each side.
    ``embed_dim`` is the per-token latent width; the conv grid size (H*W) is
    derived from ddconfig (resolution / 2**(len(ch_mult)-1)).
    """
    def __init__(self, ddconfig, lossconfig, embed_dim,
                 num_tokens=32, attn_heads=8, attn_depth=2, **kwargs):
        # The conv encoder feeds raw features to the pool (double_z off); the
        # Gaussian bottleneck is the pool's `to_moments` head.
        assert not ddconfig.get("double_z", False), \
            "ConvTokenPool pools raw conv features: set double_z: false"
        super().__init__(ddconfig, lossconfig, embed_dim, **kwargs)

        self.num_tokens = int(num_tokens)
        C_enc = ddconfig["z_channels"]
        self.C_enc = C_enc
        grid = ddconfig["resolution"] // (2 ** (len(ddconfig["ch_mult"]) - 1))
        self.grid = int(grid)                     # H == W of the conv feature map
        n_pos = self.grid * self.grid

        # encoder-side pool: K learned latents attend to the H*W conv tokens,
        # then a moments head emits (mean, logvar) per token → tiny-KL Gaussian.
        self.pool_latents = nn.Parameter(torch.randn(self.num_tokens, embed_dim) * 0.02)
        self.pool_cross = nn.ModuleList(
            [CrossAttnBlock(embed_dim, attn_heads, kv_dim=C_enc) for _ in range(attn_depth)])
        self.pool_self = nn.ModuleList(
            [SelfAttnBlock(embed_dim, attn_heads) for _ in range(attn_depth)])
        self.to_moments = nn.Linear(embed_dim, 2 * embed_dim)

        # decoder-side unpool: H*W learned position queries attend to the K latents
        self.unpool_queries = nn.Parameter(torch.randn(n_pos, embed_dim) * 0.02)
        self.unpool_cross = nn.ModuleList(
            [CrossAttnBlock(embed_dim, attn_heads) for _ in range(attn_depth)])
        self.unpool_self = nn.ModuleList(
            [SelfAttnBlock(embed_dim, attn_heads) for _ in range(attn_depth)])
        self.to_grid = nn.Linear(embed_dim, C_enc)

    # -- bottleneck ---------------------------------------------------------- #
    def _pool(self, h):
        # h: (B, C_enc, H, W) -> latent tokens (B, K, embed_dim)
        B = h.shape[0]
        tokens = h.flatten(2).transpose(1, 2)              # (B, H*W, C_enc)
        lat = self.pool_latents.unsqueeze(0).expand(B, -1, -1)
        for cross, selfb in zip(self.pool_cross, self.pool_self):
            lat = selfb(cross(lat, tokens))
        return lat

    def _unpool(self, lat):
        # lat: (B, K, embed_dim) -> conv grid (B, C_enc, H, W)
        B = lat.shape[0]
        q = self.unpool_queries.unsqueeze(0).expand(B, -1, -1)
        for cross, selfb in zip(self.unpool_cross, self.unpool_self):
            q = selfb(cross(q, lat))
        grid = self.to_grid(q)                             # (B, H*W, C_enc)
        return grid.transpose(1, 2).reshape(B, self.C_enc, self.grid, self.grid)

    # -- AutoencoderKL overrides -------------------------------------------- #
    def encode(self, x):
        lat = self._pool(self.encoder(x))                  # (B, K, embed_dim)
        moments = self.to_moments(lat)                     # (B, K, 2*embed_dim)
        moments = moments.transpose(1, 2).unsqueeze(-1)    # (B, 2*embed_dim, K, 1)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        lat = z.squeeze(-1).transpose(1, 2)                # (B, K, embed_dim)
        return self.decoder(self._unpool(lat))

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()   # (B, embed_dim, K, 1)
        dec = self.decode(z)
        if self.use_vf is not None:
            # Pool-Align only: there is no token grid to match position-wise.
            aux_feature = self.foundation_model(input)
            return dec, posterior, self.linear_proj(self._aligned_latent(posterior, z)), aux_feature
        return dec, posterior, z, None

    def _trainable_ae_params(self):
        params = (list(self.encoder.parameters()) +
                  list(self.decoder.parameters()) +
                  [self.pool_latents, self.unpool_queries] +
                  list(self.to_moments.parameters()) +
                  list(self.pool_cross.parameters()) + list(self.pool_self.parameters()) +
                  list(self.unpool_cross.parameters()) + list(self.unpool_self.parameters()) +
                  list(self.to_grid.parameters()))
        if self.linear_proj is not None:
            params += list(self.linear_proj.parameters())
        return params
