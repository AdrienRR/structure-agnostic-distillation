import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

from taming.modules.losses.vqperceptual import LPIPS, NLayerDiscriminator, weights_init, \
    adopt_weight, hinge_d_loss, vanilla_d_loss


def _dist_mean_(t: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce mean across ranks (no-op when not distributed)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t)
        t = t / dist.get_world_size()
    return t


class LPIPSWithDiscriminator(nn.Module):
    """VA-VAE tokenizer objective (L1 + LPIPS NLL with learned log-variance, KL,
    PatchGAN hinge loss) plus one optional latent-distillation term:

      - VF loss (VA-VAE):   position-wise margin cosine + within-image distance
                            matrix, on the projected latent ``z`` vs DINOv2 ``aux_feature``
      - Pool-Align:         ``vf_pooled=True``: cosine between mean-pooled projected
                            latent and mean-pooled teacher features
      - CKA / Soft-KL:      ``cka_grad_ratio > 0``: relational loss on the posterior mean

    VF and Pool-Align are gradient-norm balanced against the reconstruction (NLL)
    gradient at the encoder's last layer (``adaptive_vf``, ratio ``vf_weight``);
    the relational terms against the full VAE-loss gradient (ratio ``cka_grad_ratio``).
    """

    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_loss="hinge",
                 vf_weight=1e2, adaptive_vf=False, vf_pooled=False,
                 cos_margin=0, distmat_margin=0,
                 cka_grad_ratio=0.0, cka_grad_clamp=1e4, cka_gather=True, cka_align='cka',
                 cka_softkl_tau=0.2, cka_w_ema_decay=0.0, cka_w_rank_local=False):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        self.vf_pooled = vf_pooled
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin

        self.cka_grad_ratio = cka_grad_ratio
        self.cka_grad_clamp = cka_grad_clamp
        # True: gradient norms for the balancing are taken per rank (as in the paper
        # runs); False: probe gradients are averaged across ranks first.
        self.cka_w_rank_local = bool(cka_w_rank_local)
        self.cka_w_ema_decay = cka_w_ema_decay
        self._cka_w_ema = {}   # not checkpointed; re-measured on the first step after resume
        if cka_grad_ratio > 0:
            from ldm.modules.losses.cka import DINOCKA
            self.cka_fn = DINOCKA(gather_neighbors=cka_gather, align=cka_align, softkl_tau=cka_softkl_tau)
        else:
            self.cka_fn = None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is None:
            last_layer = self.last_layer[0]
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    def calculate_adaptive_weight_vf(self, nll_loss, vf_loss, last_layer=None):
        if last_layer is None:
            last_layer = self.last_layer[0]
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]
        vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
        vf_weight = torch.clamp(vf_weight, 0.0, 1e8).detach()
        return vf_weight * self.vf_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, split="train", z=None, aux_feature=None,
                enc_last_layer=None, z_src=None, teacher_feat=None):
        if optimizer_idx == 1:
            # discriminator update (handled first: it never needs the LPIPS forward)
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}
            return d_loss, log

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # generator update
        logits_fake = self.discriminator(reconstructions.contiguous())
        g_loss = -torch.mean(logits_fake)
        if self.disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        if z is not None and aux_feature is not None and self.vf_pooled:
            # Pool-Align: cosine between mean-pooled projected latent and mean-pooled teacher.
            zp = rearrange(z, 'b c h w -> b c (h w)').mean(-1)
            ap = rearrange(aux_feature, 'b c h w -> b c (h w)').mean(-1)
            vf_loss = torch.nn.functional.relu(
                1 - self.cos_margin - torch.nn.functional.cosine_similarity(zp, ap, dim=1)).mean()
        elif z is not None and aux_feature is not None:
            # VF loss (VA-VAE): within-image distance-matrix term + position-wise margin cosine.
            z_flat = rearrange(z, 'b c h w -> b c (h w)')
            aux_feature_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')
            z_norm = torch.nn.functional.normalize(z_flat, dim=1)
            aux_feature_norm = torch.nn.functional.normalize(aux_feature_flat, dim=1)
            z_cos_sim = torch.einsum('bci,bcj->bij', z_norm, z_norm)
            aux_feature_cos_sim = torch.einsum('bci,bcj->bij', aux_feature_norm, aux_feature_norm)
            diff = torch.abs(z_cos_sim - aux_feature_cos_sim)
            vf_loss_1 = torch.nn.functional.relu(diff - self.distmat_margin).mean()
            vf_loss_2 = torch.nn.functional.relu(
                1 - self.cos_margin - torch.nn.functional.cosine_similarity(aux_feature, z)).mean()
            vf_loss = vf_loss_1 + vf_loss_2
        else:
            vf_loss = None

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
        if vf_loss is not None:
            if self.adaptive_vf:
                try:
                    vf_weight = self.calculate_adaptive_weight_vf(nll_loss, vf_loss, last_layer=enc_last_layer)
                except RuntimeError:
                    assert not self.training
                    vf_weight = torch.tensor(0.0)
            else:
                vf_weight = self.vf_weight
            loss = loss + vf_weight * vf_loss

        # Relational distillation (CKA / Soft-KL) on the posterior mean ``z_src``,
        # gradient-norm balanced at the encoder's last layer.
        cka_terms, cka_ws = None, {}
        if self.cka_fn is not None and z_src is not None and enc_last_layer is not None:
            cka_terms = self.cka_fn.cka_loss_terms(z_src, inputs.float(), global_step=global_step,
                                                   teacher_feat=teacher_feat)   # None -> DINOv2 on the images
            g_main = torch.autograd.grad(loss, enc_last_layer, retain_graph=True)[0]
            if not self.cka_w_rank_local:
                g_main = _dist_mean_(g_main)
            g_main = g_main.norm()
            for k, term in cka_terms.items():
                g_k = torch.autograd.grad(term, enc_last_layer, retain_graph=True)[0]
                if not self.cka_w_rank_local:
                    g_k = _dist_mean_(g_k)
                g_k = g_k.norm()
                w_now = (self.cka_grad_ratio * g_main / (g_k + 1e-8)).clamp(max=self.cka_grad_clamp).detach()
                d = self.cka_w_ema_decay if k in self._cka_w_ema else 0.0
                self._cka_w_ema[k] = d * self._cka_w_ema[k] + (1.0 - d) * w_now if d > 0 else w_now
            cka_ws = dict(self._cka_w_ema)
            for k, term in cka_terms.items():
                loss = loss + cka_ws[k] * term

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/logvar".format(split): self.logvar.detach(),
               "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/nll_loss".format(split): nll_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               "{}/d_weight".format(split): d_weight.detach(),
               "{}/disc_factor".format(split): torch.tensor(disc_factor),
               "{}/g_loss".format(split): g_loss.detach().mean()}
        if posteriors is not None and hasattr(posteriors, 'std'):
            # latent-health monitors: a collapsing/inflating posterior shows up here first
            log["{}/posterior_sigma".format(split)] = posteriors.std.detach().mean()
            log["{}/latent_mu_std".format(split)] = posteriors.mode().detach().std()
        if vf_loss is not None:
            log["{}/vf_loss".format(split)] = vf_loss.detach().mean()
            log["{}/vf_weight".format(split)] = vf_weight.detach() if torch.is_tensor(vf_weight) else torch.tensor(vf_weight)
        if cka_terms is not None:
            for k, term in cka_terms.items():
                suffix = "" if k == 'main' else "_" + k
                log["{}/cka{}_loss".format(split, suffix)] = term.detach().mean()
                w = cka_ws[k]
                log["{}/cka{}_w".format(split, suffix)] = w if torch.is_tensor(w) else torch.tensor(w)
        return loss, log
