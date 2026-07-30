"""
Training script for the dual-stream forgery localization model.

Usage:
    python model/train.py                     # full training per config.yaml
    python model/train.py --smoke-test        # tiny run (train.smoke_test_epochs) for CI/sanity
    python model/train.py --resume checkpoints/last.pt

Checkpoints are written every epoch to config.train.checkpoint_dir as
both `last.pt` (always overwritten, for resuming) and `best.pt` (best
val loss so far). Mixed precision is auto-disabled on CPU-only machines.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, set_seed, get_device, save_json  # noqa: E402
from data.dataset import build_splits, make_dataloaders  # noqa: E402
from model.architecture import build_model  # noqa: E402
from model.losses import build_loss  # noqa: E402


def _make_optimizer(model, cfg):
    t_cfg = cfg["train"]
    if t_cfg["optimizer"] != "adamw":
        raise ValueError(f"Unsupported optimizer: {t_cfg['optimizer']}")
    return AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])


def _make_scheduler(optimizer, cfg, num_epochs):
    t_cfg = cfg["train"]
    if t_cfg["scheduler"] == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    return None


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, log_every=10, epoch_idx=0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0
    t0 = time.time()
    for step, (rgb, srm, mask) in enumerate(loader):
        rgb, srm, mask = rgb.to(device), srm.to(device), mask.to(device)

        with torch.set_grad_enabled(is_train):
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(rgb, srm)
                    loss, parts = loss_fn(logits, mask)
            else:
                logits = model(rgb, srm)
                loss, parts = loss_fn(logits, mask)

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if is_train and (step % log_every == 0):
            print(f"    epoch {epoch_idx} step {step}/{len(loader)} "
                  f"loss={loss.item():.4f} (bce={parts['bce']:.4f} dice={parts['dice']:.4f})")

    avg_loss = total_loss / max(1, n_batches)
    elapsed = time.time() - t0
    return avg_loss, elapsed


def train(cfg: dict, smoke_test: bool = False, resume_from: str = None):
    set_seed(cfg["project"]["seed"])
    device = get_device()
    print(f"[train] device = {device}")

    splits = build_splits(cfg)
    if len(splits["train"]) == 0:
        raise RuntimeError(
            "No training samples found. Run data/download.py then data/forge.py first."
        )

    train_loader, val_loader, _ = make_dataloaders(cfg, splits=splits, train_degrade=True)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = _make_optimizer(model, cfg)

    num_epochs = cfg["train"]["smoke_test_epochs"] if smoke_test else cfg["train"]["epochs"]
    scheduler = _make_scheduler(optimizer, cfg, num_epochs)

    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if cfg["train"]["mixed_precision"] and device.type != "cuda":
        print("[train] mixed_precision requested but no CUDA device found — running in fp32.")

    ckpt_dir = ensure_dir(resolve_path(cfg["train"]["checkpoint_dir"]))
    start_epoch = 0
    best_val_loss = float("inf")

    resume_path = resume_from or cfg["train"].get("resume_from")
    if resume_path and Path(resume_path).exists():
        print(f"[train] resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    history = []
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_time = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer, scaler=scaler,
            log_every=cfg["train"]["log_every_n_steps"], epoch_idx=epoch,
        )
        val_loss, val_time = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        if scheduler is not None:
            scheduler.step()

        print(f"[train] epoch {epoch}/{num_epochs - 1}  "
              f"train_loss={train_loss:.4f} ({train_time:.1f}s)  "
              f"val_loss={val_loss:.4f} ({val_time:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"    [train] new best val_loss={best_val_loss:.4f} -> saved best.pt")

    save_json(history, ckpt_dir / "history.json")
    print(f"[train] done. best_val_loss={best_val_loss:.4f}. "
          f"Checkpoints in {ckpt_dir}/(last.pt, best.pt)")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run only train.smoke_test_epochs epochs for a fast end-to-end check.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, smoke_test=args.smoke_test, resume_from=args.resume)
