"""
Realistic degradation pipeline, used two ways:
  1. train_time_degrade()  — random augmentation applied during training
     so the model sees phone-photo-like conditions, not pristine renders.
  2. apply_named_degradation() — a single, controllable degradation
     (fixed JPEG quality / downscale factor / moiré strength) used by
     eval/robustness.py to sweep conditions and measure the accuracy drop.

All functions operate on numpy BGR uint8 images (OpenCV convention) and
return numpy BGR uint8 images of the SAME size as the input (degradations
that change resolution internally resize back up), so masks/labels never
need to be re-aligned.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def downscale_upscale(img: np.ndarray, factor: float) -> np.ndarray:
    """Downscale then upscale back to original size (simulates a low-res
    camera capture without changing tensor dimensions downstream)."""
    if factor >= 0.999:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                        interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = max(3, int(sigma * 3) | 1)  # odd kernel size
    return cv2.GaussianBlur(img, (k, k), sigma)


def moire_overlay(img: np.ndarray, strength: float, rng: random.Random = None) -> np.ndarray:
    """Simulate screen-recapture moiré by adding a high-frequency sinusoidal
    interference pattern. strength in [0, 1]; 0 = no effect."""
    if strength <= 0:
        return img
    rng = rng or random.Random()
    h, w = img.shape[:2]
    freq_x = rng.uniform(0.15, 0.35)
    freq_y = rng.uniform(0.15, 0.35)
    angle = rng.uniform(0, np.pi)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rot_x = xx * np.cos(angle) - yy * np.sin(angle)
    rot_y = xx * np.sin(angle) + yy * np.cos(angle)
    pattern = np.sin(2 * np.pi * freq_x * rot_x) * np.sin(2 * np.pi * freq_y * rot_y)
    pattern = (pattern * 255 * strength).astype(np.float32)

    out = img.astype(np.float32) + pattern[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def train_time_degrade(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    """Apply a random combination of degradations for training-time
    augmentation, per config.yaml -> degradation.train_time."""
    d = cfg["degradation"]["train_time"]
    out = img

    lo, hi = d["downscale_range"]
    out = downscale_upscale(out, rng.uniform(lo, hi))

    if rng.random() < d["gaussian_blur_prob"]:
        slo, shi = d["gaussian_blur_sigma_range"]
        out = gaussian_blur(out, rng.uniform(slo, shi))

    qlo, qhi = d["jpeg_quality_range"]
    out = jpeg_recompress(out, rng.randint(qlo, qhi))

    return out


def apply_named_degradation(img: np.ndarray, jpeg_quality: int = None,
                             downscale_factor: float = None,
                             moire_strength: float = None,
                             rng: random.Random = None) -> np.ndarray:
    """Apply one or more explicitly-named degradations, for controlled
    eval sweeps (see eval/robustness.py)."""
    out = img
    if downscale_factor is not None:
        out = downscale_upscale(out, downscale_factor)
    if jpeg_quality is not None:
        out = jpeg_recompress(out, jpeg_quality)
    if moire_strength is not None and moire_strength > 0:
        out = moire_overlay(out, moire_strength, rng)
    return out
