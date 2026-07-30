"""
Export a visual grid of failure cases (predicted mask vs ground truth,
overlaid on the image) so failures can be inspected by eye — this is
what actually goes into the README's "Failure Analysis" section rather
than aggregate numbers alone.

Usage:
    python eval/failure_analysis.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, get_device, load_json  # noqa: E402
from data.dataset import ForgeryDataset  # noqa: E402
from eval.evaluate import load_checkpoint  # noqa: E402
from eval.metrics import pixel_iou  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _overlay(img_rgb_float: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    """img_rgb_float in [0,1], mask binary [H,W]. Returns uint8 BGR overlay."""
    img = (img_rgb_float * 255).astype(np.uint8)
    overlay = img.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def export_failures(cfg: dict, checkpoint_path: str, split_name: str = "test", n_examples: int = None):
    device = get_device()
    model = load_checkpoint(cfg, checkpoint_path, device)
    n_examples = n_examples or cfg["eval"]["num_failure_examples"]

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    samples = load_json(splits_dir / f"{split_name}.json")
    if len(samples) == 0:
        print(f"[failure_analysis] no samples in split '{split_name}' — skipping.")
        return

    ds = ForgeryDataset(samples, cfg["data"]["image_size"], degrade_fn=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    results_dir = ensure_dir(resolve_path(cfg["eval"]["results_dir"]) / "failures")
    threshold = cfg["eval"]["threshold"]

    scored = []
    with torch.no_grad():
        for i, (rgb, srm, mask) in enumerate(loader):
            if i >= len(samples):
                break
            rgb_d, srm_d = rgb.to(device), srm.to(device)
            logits = model(rgb_d, srm_d)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            gt = mask[0, 0].numpy()
            pred_bin = (prob >= threshold).astype(np.uint8)
            gt_bin = (gt >= 0.5).astype(np.uint8)
            iou = pixel_iou(pred_bin, gt_bin)
            scored.append((iou, i, rgb[0].permute(1, 2, 0).numpy(), pred_bin, gt_bin,
                           samples[i]["manipulation"]))

    # worst-IoU examples = most interesting failures
    scored.sort(key=lambda x: x[0])
    worst = scored[:n_examples]

    for rank, (iou, idx, img_rgb, pred_bin, gt_bin, manip_type) in enumerate(worst):
        gt_overlay = _overlay(img_rgb, gt_bin, color=(0, 255, 0))     # green = ground truth
        pred_overlay = _overlay(img_rgb, pred_bin, color=(0, 0, 255))  # red = predicted

        combined = np.concatenate([gt_overlay, pred_overlay], axis=1)
        out_path = results_dir / f"failure_{rank:02d}_iou{iou:.2f}_{manip_type}.png"
        cv2.imwrite(str(out_path), combined)

    print(f"[failure_analysis] wrote {len(worst)} failure-case images to {results_dir}/ "
          f"(left=ground truth in green, right=prediction in red)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    ckpt_path = resolve_path(args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[failure_analysis] checkpoint not found at {ckpt_path}. Run model/train.py first.")
        sys.exit(1)

    export_failures(cfg, str(ckpt_path), split_name=args.split, n_examples=args.n)


if __name__ == "__main__":
    main()
