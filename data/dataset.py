"""
PyTorch Dataset for the forgery segmentation task, plus a split-builder
that creates train/val/test JSON manifests under data/splits/.

Split logic:
  - All manipulation types EXCEPT config.forgery.holdout_type are pooled
    and split train/val/test by config.data.val_fraction/test_fraction.
  - The holdout_type's samples are written to a separate
    "holdout_<type>.json" manifest, used ONLY by
    eval/cross_manipulation.py to measure zero-shot generalization —
    they must never appear in train/val.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, save_json, load_json  # noqa: E402
from data.srm import compute_srm_residual  # noqa: E402


def _collect_samples(forged_root: Path, manip_type: str) -> list[dict]:
    d = forged_root / manip_type
    if not d.exists():
        return []
    samples = []
    for fake_path in sorted(d.glob("*_fake.png")):
        stem = fake_path.name[: -len("_fake.png")]
        mask_path = d / f"{stem}_mask.png"
        if mask_path.exists():
            samples.append({
                "image": str(fake_path),
                "mask": str(mask_path),
                "manipulation": manip_type,
                "label": 1,  # tampered
            })
    return samples


def build_splits(cfg: dict) -> dict[str, list[dict]]:
    """Build train/val/test/holdout manifests and write them to
    data/splits/. Returns the in-memory dict too."""
    forged_root = resolve_path(cfg["data"]["forged_dir"])
    splits_dir = ensure_dir(resolve_path(cfg["data"]["splits_dir"]))
    holdout_type = cfg["forgery"]["holdout_type"]
    all_types = cfg["forgery"]["types"]

    rng = random.Random(cfg["project"]["seed"])

    pooled = []
    holdout_samples = []
    for t in all_types:
        samples = _collect_samples(forged_root, t)
        if t == holdout_type:
            holdout_samples.extend(samples)
        else:
            pooled.extend(samples)

    rng.shuffle(pooled)
    n = len(pooled)
    val_frac = cfg["data"]["val_fraction"]
    test_frac = cfg["data"]["test_fraction"]
    n_val = max(1, int(n * val_frac)) if n > 0 else 0
    n_test = max(1, int(n * test_frac)) if n > 0 else 0
    n_train = max(0, n - n_val - n_test)

    train = pooled[:n_train]
    val = pooled[n_train:n_train + n_val]
    test = pooled[n_train + n_val: n_train + n_val + n_test]

    save_json(train, splits_dir / "train.json")
    save_json(val, splits_dir / "val.json")
    save_json(test, splits_dir / "test.json")
    save_json(holdout_samples, splits_dir / f"holdout_{holdout_type}.json")

    print(f"[dataset] splits -> train={len(train)} val={len(val)} test={len(test)} "
          f"holdout({holdout_type})={len(holdout_samples)}")
    return {"train": train, "val": val, "test": test, f"holdout_{holdout_type}": holdout_samples}


class ForgeryDataset(Dataset):
    """Returns (rgb_tensor [3,H,W], srm_tensor [3,H,W], mask_tensor [1,H,W])."""

    def __init__(self, samples: list[dict], image_size: int, degrade_fn=None, rng_seed: int = 0):
        self.samples = samples
        self.image_size = image_size
        self.degrade_fn = degrade_fn  # optional callable(img_bgr, rng) -> img_bgr
        self.rng = random.Random(rng_seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)

        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.degrade_fn is not None:
            img = self.degrade_fn(img, self.rng)

        srm = compute_srm_residual(img)

        rgb_t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        srm_t = torch.from_numpy(srm).permute(2, 0, 1).float()
        mask_bin = (mask > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0).float()

        return rgb_t, srm_t, mask_t


def make_dataloaders(cfg: dict, splits: dict[str, list[dict]] = None, train_degrade=True):
    from torch.utils.data import DataLoader
    from functools import partial
    from data.degrade import train_time_degrade

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    if splits is None:
        splits = {
            "train": load_json(splits_dir / "train.json"),
            "val": load_json(splits_dir / "val.json"),
            "test": load_json(splits_dir / "test.json"),
        }

    image_size = cfg["data"]["image_size"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]

    degrade_fn = None
    if train_degrade:
        def degrade_fn(img, rng):
            return train_time_degrade(img, cfg, rng)

    train_ds = ForgeryDataset(splits["train"], image_size, degrade_fn=degrade_fn, rng_seed=cfg["project"]["seed"])
    val_ds = ForgeryDataset(splits["val"], image_size, degrade_fn=None)
    test_ds = ForgeryDataset(splits["test"], image_size, degrade_fn=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    cfg = load_config()
    build_splits(cfg)
