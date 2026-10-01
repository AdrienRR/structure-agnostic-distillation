import json
import os

import numpy as np
import pytorch_lightning as pl
import torch

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


class AutoencoderKL(pl.LightningModule):
    """VA-VAE convolutional KL autoencoder with optional latent distillation.

    use_vf='dinov2' loads a frozen DINOv2 teacher for the VF loss (position-wise)
    or Pool-Align (``lossconfig.params.vf_pooled``). Both map the latent into the
    teacher's feature space with a learned bias-free linear map, ``linear_proj``.
    text_teacher_root points to precomputed caption embeddings
    (tools/precompute_text_embeddings_bge.py), used instead of DINOv2.
    CKA / Soft-KL are configured entirely in the loss (``cka_grad_ratio``).
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 use_vf=None,
                 optim_betas=(0.5, 0.9),
                 text_teacher_root=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        # Cross-modal teacher: one precomputed caption embedding per training image,
        # memory-mapped so all DDP ranks on a node share the page cache.
        self.text_teacher_root = text_teacher_root
        if text_teacher_root is not None:
            self._text_emb = np.load(os.path.join(text_teacher_root, "embeddings_fp16.npy"), mmap_mode="r")
            self._text_key_to_index = json.load(open(os.path.join(text_teacher_root, "key_to_index.json")))
            print(f"[text-teacher] {self._text_emb.shape} embeddings from {text_teacher_root}")
        if ddconfig["double_z"]:
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        else:
            # only reached by subclasses that replace encode(); kept so that the
            # module layout (and checkpoints) match
            self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.optim_betas = tuple(optim_betas)
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # VF and Pool-Align compare latent and teacher features directly, so they learn a
        # bias-free linear map W from the latent into the teacher's space (a 1x1 conv, i.e.
        # the same W at every position; for Pool-Align, W commutes with the mean-pooling).
        # The relational objectives compare B x B similarity matrices and need no projector.
        self.use_vf = use_vf
        self.linear_proj = None
        teacher_dim = None
        if use_vf is not None:
            from ldm.models.foundation_models import aux_foundation_model
            print(f"Using {use_vf} as auxiliary feature.")
            self.foundation_model = aux_foundation_model(use_vf)
            teacher_dim = self.foundation_model.feature_dim
        elif text_teacher_root is not None and getattr(self.loss, "vf_pooled", False):
            teacher_dim = self._text_emb.shape[1]
        if teacher_dim is not None:
            self.linear_proj = torch.nn.Conv2d(embed_dim, teacher_dim, kernel_size=1, bias=False)
        self.automatic_optimization = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def _aligned_latent(self, posterior, z):
        """Latent the VF / Pool-Align term acts on. VF keeps VA-VAE's choice, the posterior
        sample; Pool-Align, like the relational objectives, uses the posterior mean, since
        matching the sample lets the encoder satisfy the target by inflating the variance."""
        return posterior.mode() if self.loss.vf_pooled else z

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        if self.use_vf is not None:
            aux_feature = self.foundation_model(input)
            z = self.linear_proj(self._aligned_latent(posterior, z))
            if z.shape[-2:] != aux_feature.shape[-2:]:
                z = torch.nn.functional.interpolate(z, size=aux_feature.shape[-2:], mode='bilinear',
                                                    align_corners=False)
            return dec, posterior, z, aux_feature
        return dec, posterior, z, None

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    def _lookup_text_teacher(self, batch, ref):
        """Caption embedding of each image, keyed by file basename (n{wnid}_{imgid})."""
        ids = [os.path.splitext(os.path.basename(p))[0] for p in batch["file_path_"]]
        rows = [self._text_key_to_index[i] for i in ids]
        emb = np.asarray(self._text_emb[rows], dtype=np.float32)   # (B, D)
        return torch.from_numpy(emb).to(ref.device)

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, z, aux_feature = self(inputs)
        ae_opt, disc_opt = self.optimizers()
        enc_last_layer = self.encoder.conv_out.weight
        # The relational target is the posterior MODE: matching the reparameterised
        # sample lets the encoder inflate the posterior variance to satisfy the
        # target trivially, collapsing reconstruction.
        z_src = z if posterior is None else posterior.mode()
        teacher_feat = None
        if self.text_teacher_root is not None:
            temb = self._lookup_text_teacher(batch, z_src)          # (B, D)
            if self.linear_proj is not None:
                # Pool-Align: W maps the posterior mean into the caption-embedding space;
                # the caption embedding is the teacher's (already pooled) descriptor.
                z = self.linear_proj(z_src)
                aux_feature = temb[:, :, None, None]
            else:
                teacher_feat = temb                                  # CKA / Soft-KL
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train", z=z,
                                        aux_feature=aux_feature, enc_last_layer=enc_last_layer,
                                        z_src=z_src, teacher_feat=teacher_feat)
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        self.clip_gradients(ae_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        ae_opt.step()

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            enc_last_layer=enc_last_layer)
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0, data_type=None):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, z, aux_feature = self(inputs)
        enc_last_layer = self.encoder.conv_out.weight
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", z=z,
                                        aux_feature=aux_feature, enc_last_layer=enc_last_layer)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",
                                            enc_last_layer=enc_last_layer)
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def _trainable_ae_params(self):
        params = (list(self.encoder.parameters()) + list(self.decoder.parameters()) +
                  list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters()))
        if self.linear_proj is not None:
            params += list(self.linear_proj.parameters())
        return params

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self._trainable_ae_params(), lr=lr, betas=self.optim_betas)
        disc_params = [p for p in self.loss.discriminator.parameters() if p.requires_grad]
        opt_disc = torch.optim.Adam(disc_params, lr=lr, betas=self.optim_betas)
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        if not only_inputs:
            xrec, posterior, z, _ = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
