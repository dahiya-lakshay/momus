"""
SRM (Spatial Rich Model) noise-residual filters.

These are the classic high-pass filter kernels from steganalysis /
image-forensics literature (Fridrich & Kodovsky, 2012) used widely in
forgery-detection papers (e.g. the "RGB-N" and "SPAN" model families)
to give the network an explicit noise-residual view of the image,
since tampered regions often leave faint high-frequency inconsistencies
that are invisible in raw RGB but pop out after high-pass filtering.

We apply 3 fixed 5x5 SRM kernels (one per output channel) to the
grayscale image and stack them into a 3-channel "SRM residual image"
that is fed to the model's second input branch.
"""
from __future__ import annotations

import numpy as np
import cv2

# Three standard SRM high-pass kernels (5x5), normalized.
_SRM_KERNELS = [
    np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0,
    np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0,
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 2.0,
]


def compute_srm_residual(img_bgr: np.ndarray) -> np.ndarray:
    """Compute a 3-channel SRM residual map from a BGR uint8 image.
    Returns float32 array in roughly [-1, 1] after clipping+normalizing,
    same H x W as input, 3 channels (one per kernel)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    channels = []
    for kernel in _SRM_KERNELS:
        residual = cv2.filter2D(gray, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        residual = np.clip(residual, -8, 8) / 8.0  # normalize to [-1, 1]
        channels.append(residual)
    return np.stack(channels, axis=-1).astype(np.float32)
