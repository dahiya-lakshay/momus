"""
Dual-stream segmentation model for tamper localization.

Two encoder branches:
  - RGB stream: sees the raw image, learns semantic/appearance cues
    (e.g. a photo box that looks pasted-in, inconsistent lighting).
  - SRM stream: sees the noise-residual map (data/srm.py), which
    highlights high-frequency inconsistencies invisible in raw RGB —
    the classic forensics signal for copy-move/splicing/inpainting.

Features from both streams are concatenated at the bottleneck
("encoder_input" fusion point per config.yaml) and passed through a
shared U-Net-style decoder with skip connections from the RGB encoder,
outputting a single-channel tamper-probability mask at full resolution.

This is a lightweight, from-scratch implementation (no pretrained
weights required) so the whole pipeline is guaranteed to run offline.
If model.name == "segformer_b0" in config.yaml, an alternate HF-backed
encoder is used instead (see build_model()) — that path DOES require
downloading pretrained weights from Hugging Face, which needs internet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Simple U-Net style encoder. Returns bottleneck feature + skip list."""

    def __init__(self, in_channels: int, base_channels: int, depth: int):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        out_ch = base_channels
        for i in range(depth):
            self.blocks.append(conv_block(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
            out_ch *= 2
        self.out_channels = ch  # channels at the deepest block, before final pool

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            x = self.pools[i](x)
        return x, skips


class DualStreamUNet(nn.Module):
    """RGB + SRM dual-stream encoder, fused bottleneck, U-Net decoder."""

    def __init__(self, in_channels_rgb: int = 3, in_channels_srm: int = 3,
                 base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.rgb_encoder = Encoder(in_channels_rgb, base_channels, depth)
        self.srm_encoder = Encoder(in_channels_srm, base_channels, depth)

        fused_ch = self.rgb_encoder.out_channels + self.srm_encoder.out_channels
        self.bottleneck = conv_block(fused_ch, fused_ch)

        # Decoder: upsample + concat RGB-stream skip + conv, mirroring encoder depth
        decoder_in_ch = fused_ch
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        skip_channels = [base_channels * (2 ** i) for i in range(depth)][::-1]
        ch = decoder_in_ch
        for skip_ch in skip_channels:
            self.up_blocks.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(conv_block(skip_ch + skip_ch, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor) -> torch.Tensor:
        rgb_feat, rgb_skips = self.rgb_encoder(rgb)
        srm_feat, _srm_skips = self.srm_encoder(srm)

        x = torch.cat([rgb_feat, srm_feat], dim=1)
        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(rgb_skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        logits = self.head(x)  # [B, 1, H, W], raw logits (apply sigmoid outside)
        return logits


class CombinedInputWrapper(nn.Module):
    """Wraps DualStreamUNet to accept a single concatenated 6-channel
    tensor [rgb(3) || srm(3)] as input — used for ONNX export, since
    ONNX/onnxruntime handle a single input tensor more simply than two."""

    def __init__(self, model: DualStreamUNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, srm = x[:, :3], x[:, 3:6]
        logits = self.model(rgb, srm)
        return torch.sigmoid(logits)


def build_model(cfg: dict) -> nn.Module:
    m_cfg = cfg["model"]
    if m_cfg["name"] == "segformer_b0":
        return _build_segformer(cfg)
    return DualStreamUNet(
        in_channels_rgb=m_cfg["in_channels_rgb"],
        in_channels_srm=m_cfg["in_channels_srm"],
        base_channels=m_cfg["base_channels"],
        depth=m_cfg["depth"],
    )


def _build_segformer(cfg: dict) -> nn.Module:
    """Optional alternate encoder using HuggingFace SegFormer-B0.
    Requires internet access to download pretrained weights and the
    `transformers` package. Falls back to DualStreamUNet with a clear
    warning if unavailable, so run_all.sh never hard-fails on this path.
    """
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        print("[architecture] transformers not installed — falling back to DualStreamUNet.")
        m_cfg = cfg["model"]
        return DualStreamUNet(m_cfg["in_channels_rgb"], m_cfg["in_channels_srm"],
                               m_cfg["base_channels"], m_cfg["depth"])

    class SegformerWrapper(nn.Module):
        """Adapts a pretrained SegFormer-B0 (ImageNet-style 3-channel
        input) to our dual-stream setup by summing RGB and SRM into a
        single 3-channel input via a learned 1x1 mix, then running the
        pretrained backbone. Less principled than the true dual encoder
        but lets you leverage pretrained ImageNet features if you have
        internet access for the initial download."""

        def __init__(self):
            super().__init__()
            self.mixer = nn.Conv2d(6, 3, kernel_size=1)
            try:
                self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=1,
                    ignore_mismatched_sizes=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not download SegFormer-B0 weights ({e}). "
                    f"Set model.name: dual_stream_unet in config.yaml to run offline."
                )

        def forward(self, rgb, srm):
            x = self.mixer(torch.cat([rgb, srm], dim=1))
            out = self.backbone(pixel_values=x).logits
            out = F.interpolate(out, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            return out

    return SegformerWrapper()
