import os, sys
import torch
from omegaconf import OmegaConf
from torchvision import transforms

# vavae/ subdirectory must be on path for ldm imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vavae'))
from ldm.util import instantiate_from_config


class VA_VAE:
    """Frozen trained tokenizer for latent extraction and decoding.

    The config gives `ckpt_path` (Lightning checkpoint written by vavae/main.py) and
    `model_config` (the tokenizer training config it was trained with)."""

    def __init__(self, config, img_size=256):
        cfg = OmegaConf.load(config)
        self.img_size = img_size
        self.ckpt_path = cfg.ckpt_path
        self.model_config_path = cfg.model_config

        model_cfg = OmegaConf.load(self.model_config_path)
        # Encode/decode need neither the training loss (LPIPS, discriminator,
        # distillation teachers) nor the teacher projections: build without them.
        model_cfg.model.params.lossconfig = OmegaConf.create({"target": "torch.nn.Identity"})
        model_cfg.model.params.use_vf = None
        model_cfg.model.params.text_teacher_root = None
        self.model = instantiate_from_config(model_cfg.model)
        sd = torch.load(self.ckpt_path, map_location='cpu')['state_dict']
        sd = {k: v for k, v in sd.items()
              if not k.startswith(('loss.', 'foundation_model.', 'linear_proj.'))}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"checkpoint mismatch: {missing[:5]} {unexpected[:5]}"
        self.model = self.model.cuda().eval()

    def img_transform(self, p_hflip=0.0, img_size=None):
        sz = img_size or self.img_size
        return transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def encode_images(self, images):
        with torch.no_grad():
            z = self.model.encode(images.cuda()).sample()   # posterior sample, as in VA-VAE
            return z.contiguous()                           # safetensors needs contiguous tensors

    def decode_to_images(self, z):
        with torch.no_grad():
            images = self.model.decode(z.cuda())
            images = torch.clamp(127.5 * images + 128.0, 0, 255)
            return images.permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()
