"""Frozen vision foundation model used as the VF / Pool-Align teacher.

Adapted from VA-VAE (Jingfeng Yao, HUST-VL).
"""
import timm
import torch
import torch.nn as nn


def get_dinov2_encoder():
    """DINOv2 ViT-L/14 from timm, frozen."""
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


class aux_foundation_model(nn.Module):
    """Returns the teacher's patch features as a (B, 1024, h/16, w/16) map."""

    def __init__(self, type):
        super().__init__()
        assert type == 'dinov2', f"Unsupported foundation model type: {type}"
        self.model = get_dinov2_encoder()
        self.type = type
        self.feature_dim = 1024

    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model.forward_features(x)[:, 1:].reshape(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)

    def forward(self, x):
        with torch.no_grad():
            return self.forward_dinov2(x)
