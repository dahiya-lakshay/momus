"""
Cross-manipulation generalization eval.

The model is trained on all manipulation types EXCEPT
config.forgery.holdout_type (default: inpaint_removal — see
data/dataset.py build_splits()). This script evaluates the SAME
checkpoint on:
    (a) the in-distribution test split (types seen during training)
    (b) the held-out manipulation type (never seen during training)

...and reports the metric GAP between them. A large gap means the
model learned type-specific shortcuts rather than a general notion of
"this region looks tampered" — exactly the kind of thing to flag
honestly in the README's Failure Analysis section rather than hide.

Usage:
    python eval/cross_manipulation.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, save_json, markdown_table, load_json  # noqa: E402
from eval.evaluate import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = ensure_dir(resolve_path(cfg["eval"]["results_dir"]))
    holdout_type = cfg["forgery"]["holdout_type"]

    ckpt_path = resolve_path(args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[cross_manipulation] checkpoint not found at {ckpt_path}. Run model/train.py first.")
        sys.exit(1)

    in_dist = evaluate(cfg, str(ckpt_path), split_name="test")

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    holdout_samples = load_json(splits_dir / f"holdout_{holdout_type}.json")
    holdout = evaluate(cfg, str(ckpt_path), split_name=f"holdout_{holdout_type}", samples=holdout_samples)

    if in_dist is None or holdout is None:
        print("[cross_manipulation] insufficient samples in one of the splits — skipping gap report.")
        return

    iou_gap = in_dist["pixel_iou"] - holdout["pixel_iou"]
    f1_gap = in_dist["pixel_f1"] - holdout["pixel_f1"]

    print(f"[cross_manipulation] in-distribution IoU={in_dist['pixel_iou']:.4f}  "
          f"holdout({holdout_type}) IoU={holdout['pixel_iou']:.4f}  gap={iou_gap:+.4f}")

    result = {
        "holdout_type": holdout_type,
        "in_distribution": in_dist,
        "holdout": holdout,
        "iou_gap": iou_gap,
        "f1_gap": f1_gap,
    }
    save_json(result, results_dir / "cross_manipulation.json")

    table = markdown_table(
        headers=["Eval set", "N", "Pixel IoU", "Pixel F1"],
        rows=[
            ["In-distribution (test)", in_dist["n_samples"], f"{in_dist['pixel_iou']:.4f}", f"{in_dist['pixel_f1']:.4f}"],
            [f"Holdout ({holdout_type}, zero-shot)", holdout["n_samples"], f"{holdout['pixel_iou']:.4f}", f"{holdout['pixel_f1']:.4f}"],
            ["Generalization gap (in-dist − holdout)", "-", f"{iou_gap:+.4f}", f"{f1_gap:+.4f}"],
        ],
    )
    md = (f"# Cross-Manipulation Generalization\n\n"
          f"Trained on all manipulation types except **{holdout_type}**, "
          f"which is evaluated here zero-shot.\n\n{table}\n\n"
          f"> A large positive gap means the model relies on shortcuts specific "
          f"to the manipulation types it was trained on, rather than a general "
          f"tamper-detection signal. Fill in an interpretation here after a real "
          f"(non-smoke-test) training run.\n")
    (results_dir / "cross_manipulation.md").write_text(md)
    print(f"[cross_manipulation] wrote {results_dir / 'cross_manipulation.md'}")


if __name__ == "__main__":
    main()
