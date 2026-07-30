"""
Main evaluation entrypoint. Loads a checkpoint, runs it over the test
split, computes pixel IoU/F1 and image-level AUC, and writes:
    results/eval_summary.md   (markdown table)
    results/eval_summary.json (raw numbers, used by other scripts)

Usage:
    python eval/evaluate.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, get_device, save_json, markdown_table  # noqa: E402
from data.dataset import make_dataloaders, ForgeryDataset  # noqa: E402
from utils.common import load_json  # noqa: E402
from model.architecture import build_model  # noqa: E402
from eval.metrics import batch_pixel_metrics, image_level_auc  # noqa: E402


def load_checkpoint(cfg: dict, checkpoint_path: str, device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, threshold: float):
    all_probs, all_gts = [], []
    for rgb, srm, mask in loader:
        rgb, srm = rgb.to(device), srm.to(device)
        logits = model(rgb, srm)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # [B, H, W]
        gts = mask.numpy()[:, 0]
        all_probs.append(probs)
        all_gts.append(gts)
    return np.concatenate(all_probs, axis=0), np.concatenate(all_gts, axis=0)


def evaluate(cfg: dict, checkpoint_path: str, split_name: str = "test", samples=None):
    device = get_device()
    model = load_checkpoint(cfg, checkpoint_path, device)

    if samples is None:
        splits_dir = resolve_path(cfg["data"]["splits_dir"])
        samples = load_json(splits_dir / f"{split_name}.json")

    if len(samples) == 0:
        print(f"[evaluate] WARNING: 0 samples in split '{split_name}' — skipping.")
        return None

    ds = ForgeryDataset(samples, cfg["data"]["image_size"], degrade_fn=None)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    probs, gts = run_inference(model, loader, device, cfg["eval"]["threshold"])
    pix_metrics = batch_pixel_metrics(probs, gts, threshold=cfg["eval"]["threshold"])

    # All samples here are tampered (label=1); image-level AUC needs both
    # classes to be meaningful. We treat "fraction of pixels predicted
    # tampered above threshold" > 0 as a proxy detection rate here, and
    # separately report AUC as NaN with an explanation if only one class
    # is present in this split.
    image_labels = np.ones(len(samples))  # this dataset variant is forged-only per split
    auc = image_level_auc(probs, image_labels)

    result = {
        "split": split_name,
        "n_samples": len(samples),
        "pixel_iou": pix_metrics["iou"],
        "pixel_f1": pix_metrics["f1"],
        "image_auc": auc,
        "auc_note": ("NaN because this split contains only tampered images; "
                     "image-level AUC needs a mix of real+fake images. See "
                     "eval/cross_manipulation.py or add real (untampered) "
                     "samples to the split for a meaningful AUC.") if np.isnan(auc) else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = ensure_dir(resolve_path(cfg["eval"]["results_dir"]))

    ckpt_path = resolve_path(args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[evaluate] checkpoint not found at {ckpt_path}. Run model/train.py first.")
        sys.exit(1)

    result = evaluate(cfg, str(ckpt_path), split_name=args.split)
    if result is None:
        sys.exit(0)

    print(f"[evaluate] split={result['split']}  n={result['n_samples']}  "
          f"IoU={result['pixel_iou']:.4f}  F1={result['pixel_f1']:.4f}  "
          f"image_AUC={result['image_auc']}")

    save_json(result, results_dir / "eval_summary.json")

    table = markdown_table(
        headers=["Split", "N", "Pixel IoU", "Pixel F1", "Image-level AUC"],
        rows=[[result["split"], result["n_samples"],
               f"{result['pixel_iou']:.4f}", f"{result['pixel_f1']:.4f}",
               "N/A (forged-only split)" if result["auc_note"] else f"{result['image_auc']:.4f}"]],
    )
    md = f"# Evaluation Summary\n\n{table}\n"
    if result["auc_note"]:
        md += f"\n> **Note:** {result['auc_note']}\n"
    (results_dir / "eval_summary.md").write_text(md)
    print(f"[evaluate] wrote {results_dir / 'eval_summary.md'}")


if __name__ == "__main__":
    main()
