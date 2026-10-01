"""Relational (structure-agnostic) distillation objectives.

Each image's tokens are pooled into one L2-normalised descriptor, and the B x B
matrix of between-image cosine similarities of the student latent is matched to
that of a frozen teacher, over the all-gathered global batch:

  - ``align='cka'``    : 1 - unbiased linear CKA of the two matrices (global geometry)
  - ``align='softkl'`` : per-image KL between softmaxed neighbour distributions
                          (local geometry, temperature ``softkl_tau``)

The teacher is a frozen DINOv2 ViT-L/14, or a precomputed per-image descriptor
(e.g. a caption embedding) passed as ``teacher_feat``.
"""
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ───────────────────────── distributed gather helpers ───────────────────── #

def all_gather_nograd(t: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks, returning a detached concatenation."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t.detach()
    # NCCL rejects non-contiguous buffers, and empty_like preserves t's layout.
    t_c = t.contiguous()
    gathered = [torch.empty_like(t_c) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t_c)
    return torch.cat(gathered, dim=0)


class _GatherGrad(torch.autograd.Function):
    """Differentiable all-gather. Backward returns this rank's slice of the gradient,
    scaled by world_size so that DDP's gradient averaging recovers the true gradient
    of the global (batch-level) loss."""

    @staticmethod
    def forward(ctx, t):
        ctx.ws = dist.get_world_size()
        ctx.rank = dist.get_rank()
        t_c = t.contiguous()
        out = [torch.empty_like(t_c) for _ in range(ctx.ws)]
        dist.all_gather(out, t_c)
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad):
        bs = grad.shape[0] // ctx.ws
        return (ctx.ws * grad[ctx.rank * bs:(ctx.rank + 1) * bs]).contiguous()


def all_gather_grad(t: torch.Tensor) -> torch.Tensor:
    """All-gather that preserves gradient to the local rank's contribution."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t
    return _GatherGrad.apply(t)


# ───────────────────────────── relational losses ────────────────────────── #

def pooled_kernel(feats: torch.Tensor) -> torch.Tensor:
    """(B, N, D) -> (B, B): cosine between per-image means of L2-normalised tokens.
    Invariant to the order, number and layout of tokens."""
    p = F.normalize(F.normalize(feats, dim=-1).mean(dim=1), dim=-1)   # (B, D)
    return p @ p.t()


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al., 2012) on zero-diagonal kernels. Requires n >= 4."""
    n = K.shape[0]
    Kt = K - torch.diag(torch.diagonal(K))
    Lt = L - torch.diag(torch.diagonal(L))
    one = torch.ones(n, device=K.device, dtype=K.dtype)
    t1 = (Kt * Lt).sum()
    t2 = (one @ Kt @ one) * (one @ Lt @ one) / ((n - 1) * (n - 2))
    t3 = (2.0 / (n - 2)) * (one @ Kt @ Lt @ one)
    return (t1 + t2 - t3) / (n * (n - 3))


def linear_cka_unbiased(K: torch.Tensor, L: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Linear CKA on the unbiased HSIC estimator. The self-terms can go slightly
    negative on near-constant kernels (e.g. at initialisation), so they are floored
    at ``eps`` and the result is clamped to the valid range [-1, 1]."""
    hkk = hsic_unbiased(K, K).clamp_min(eps)
    hll = hsic_unbiased(L, L).clamp_min(eps)
    return (hsic_unbiased(K, L) / torch.sqrt(hkk * hll)).clamp(-1.0, 1.0)


def neighbor_softkl(M_z: torch.Tensor, M_d: torch.Tensor,
                    tau: float = 0.2, eps: float = 1e-9) -> torch.Tensor:
    """Mean over images of KL(teacher || student) between the softmaxed rows of the
    two similarity matrices (self-similarity masked out, temperature ``tau``)."""
    n = M_z.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=M_z.device)

    def rowdist(M):
        return torch.softmax(M.masked_fill(~off, float('-inf')) / tau, dim=1)
    pz = rowdist(M_z)
    pd = rowdist(M_d)
    return (pd * (torch.log(pd + eps) - torch.log(pz + eps))).sum(dim=1).mean()


def relational_loss(feats_z: torch.Tensor, feats_t: torch.Tensor, gather: bool = True,
                    align: str = 'cka', softkl_tau: float = 0.2) -> torch.Tensor:
    """feats_z: (B, N_z, C) student tokens (with grad); feats_t: (B, N_t, D) teacher tokens."""
    if gather:
        feats_z = all_gather_grad(feats_z)
        feats_t = all_gather_nograd(feats_t)
    M_z = pooled_kernel(feats_z.float())
    M_t = pooled_kernel(feats_t.float()).detach()
    if align == 'softkl':
        return neighbor_softkl(M_z, M_t, tau=softkl_tau)
    if align == 'cka':
        return 1.0 - linear_cka_unbiased(M_z, M_t)
    raise ValueError(f"unknown align {align!r} (expected 'cka' or 'softkl')")


# ─────────────────────────────── teacher module ─────────────────────────── #

class DINOCKA(nn.Module):
    """Relational distillation loss with a frozen DINOv2 teacher (timm)."""

    def __init__(self,
                 model_name: str = 'vit_large_patch14_dinov2.lvd142m',
                 target_size: int = 224,
                 gather_neighbors: bool = True,
                 align: str = 'cka',
                 softkl_tau: float = 0.2):
        super().__init__()
        self.align = align
        self.softkl_tau = softkl_tau
        self.dino = timm.create_model(f'hf-hub:timm/{model_name}', pretrained=True, dynamic_img_size=True)
        self.dino.eval()
        self.dino.requires_grad_(False)
        self.target_size = target_size
        self.gather_neighbors = gather_neighbors
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> ImageNet-normalised, longest side resized to ``target_size``."""
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
        if self.target_size is not None:
            H, W = x.shape[2], x.shape[3]
            long_side = max(H, W)
            if long_side != self.target_size:
                scale = self.target_size / long_side
                x = F.interpolate(x, size=(max(1, round(H * scale)), max(1, round(W * scale))),
                                  mode='bicubic', align_corners=False)
        return (x - self.mean) / self.std

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) DINOv2 patch tokens (class token dropped), computed in fp32."""
        with torch.autocast('cuda', enabled=False):
            tokens = self.dino.forward_features(self._preprocess(x.float()))
        return tokens[:, 1:]

    def cka_loss_terms(self, z: torch.Tensor, x: torch.Tensor, global_step=None,
                       teacher_feat: torch.Tensor = None) -> dict:
        """z: (B, C, h, w) latent; x: (B, 3, H, W) image in [-1, 1].
        teacher_feat: optional (B, D) precomputed teacher descriptor (text teacher);
        when given, the DINOv2 forward is skipped."""
        if teacher_feat is not None:
            feats_t = teacher_feat.unsqueeze(1)                  # (B, 1, D)
        else:
            with torch.no_grad():
                feats_t = self._patches(x.float())
        feats_z = z.flatten(2).transpose(1, 2)                   # (B, N_z, C)
        return {'main': relational_loss(feats_z, feats_t, self.gather_neighbors,
                                        align=self.align, softkl_tau=self.softkl_tau)}
