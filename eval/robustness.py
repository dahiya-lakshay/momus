"""
Robustness sweep. Applies controlled degradations (JPEG quality,
downscale factor, simulated screen-recapture moiré) to the test set at
eval time and measures how pixel IoU degrades, per config.yaml ->
degradation.eval_sweep. Writes:
    results/robustness_jpeg.png / .md
    results/robustness_downscale.png / .md
    results/robustness_moire.png / .md
    results/robustness_summary.json

Usage:
    python eval/robustness.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, get_device, save_json, markdown_table, load_json  # noqa: E402
from data.dataset import ForgeryDataset  # noqa: E402
from data.degrade import apply_named_degradation  # noqa: E402
from eval.evaluate import load_checkpoint, run_inference  # noqa: E402
from eval.metrics import batch_pixel_metrics  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _eval_with_degradation(model, samples, cfg, device, degrade_kwargs):
    rng = random.Random(cfg["project"]["seed"])

    def degrade_fn(img, _rng):
        return apply_named_degradation(img, rng=rng, **degrade_kwargs)

    ds = ForgeryDataset(samples, cfg["data"]["image_size"], degrade_fn=degrade_fn)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
    probs, gts = run_inference(model, loader, device, cfg["eval"]["threshold"])
    return batch_pixel_metrics(probs, gts, threshold=cfg["eval"]["threshold"])


def _plot_sweep(x_values, y_values, xlabel, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Pixel IoU")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_robustness_sweep(cfg: dict, checkpoint_path: str):
    device = get_device()
    model = load_checkpoint(cfg, checkpoint_path, device)

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    test_samples = load_json(splits_dir / "test.json")
    if len(test_samples) == 0:
        print("[robustness] no test samples found — skipping.")
        return None

    sweep_cfg = cfg["degradation"]["eval_sweep"]
    results_dir = ensure_dir(resolve_path(cfg["eval"]["results_dir"]))
    summary = {}

    # --- JPEG quality sweep (downscale/moire held at "no degradation") ---
    jpeg_results = []
    for q in sweep_cfg["jpeg_qualities"]:
        m = _eval_with_degradation(model, test_samples, cfg, device, {"jpeg_quality": q})
        jpeg_results.append({"jpeg_quality": q, **m})
        print(f"[robustness] jpeg_quality={q:>3}  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")
    summary["jpeg"] = jpeg_results
    _plot_sweep([r["jpeg_quality"] for r in jpeg_results], [r["iou"] for r in jpeg_results],
                "JPEG quality", "Robustness vs JPEG recompression", results_dir / "robustness_jpeg.png")

    # --- Downscale sweep ---
    downscale_results = []
    for f in sweep_cfg["downscale_factors"]:
        m = _eval_with_degradation(model, test_samples, cfg, device, {"downscale_factor": f})
        downscale_results.append({"downscale_factor": f, **m})
        print(f"[robustness] downscale_factor={f:.2f}  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")
    summary["downscale"] = downscale_results
    _plot_sweep([r["downscale_factor"] for r in downscale_results], [r["iou"] for r in downscale_results],
                "Downscale factor (1.0 = original res)", "Robustness vs resolution loss",
                results_dir / "robustness_downscale.png")

    # --- Moiré (simulated screen recapture) sweep ---
    moire_results = []
    for s in sweep_cfg["moiré_strengths"]:
        m = _eval_with_degradation(model, test_samples, cfg, device, {"moire_strength": s})
        moire_results.append({"moire_strength": s, **m})
        print(f"[robustness] moire_strength={s:.2f}  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")
    summary["moire"] = moire_results
    _plot_sweep([r["moire_strength"] for r in moire_results], [r["iou"] for r in moire_results],
                "Moiré strength (simulated screen recapture)", "Robustness vs screen-recapture artifacts",
                results_dir / "robustness_moire.png")

    save_json(summary, results_dir / "robustness_summary.json")

    md_parts = ["# Robustness Sweep\n"]
    for name, rows, key in [("JPEG recompression", jpeg_results, "jpeg_quality"),
                              ("Downscale / resolution loss", downscale_results, "downscale_factor"),
                              ("Simulated screen-recapture moiré", moire_results, "moire_strength")]:
        table = markdown_table(
            headers=[key, "Pixel IoU", "Pixel F1"],
            rows=[[r[key], f"{r['iou']:.4f}", f"{r['f1']:.4f}"] for r in rows],
        )
        md_parts.append(f"\n## {name}\n\n{table}\n")
    (results_dir / "robustness_summary.md").write_text("\n".join(md_parts))
    print(f"[robustness] wrote {results_dir}/robustness_*.png and robustness_summary.md")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    ckpt_path = resolve_path(args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[robustness] checkpoint not found at {ckpt_path}. Run model/train.py first.")
        sys.exit(1)

    run_robustness_sweep(cfg, str(ckpt_path))


if __name__ == "__main__":
    main()
