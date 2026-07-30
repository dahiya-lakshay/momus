"""
Export a trained checkpoint to ONNX for production deployment.

Uses CombinedInputWrapper (model/architecture.py) so the exported
graph takes a single 6-channel input [rgb(3) || srm(3)] instead of two
separate tensors — simpler for downstream ONNX Runtime serving code,
and it's what deploy/benchmark.py and deploy/quantize.py expect.

Usage:
    python deploy/export_onnx.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir  # noqa: E402
from model.architecture import build_model, CombinedInputWrapper  # noqa: E402


def export(cfg: dict, checkpoint_path: str, output_path: str = None):
    device = torch.device("cpu")  # export from CPU for portability
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapped = CombinedInputWrapper(model).eval()

    image_size = cfg["data"]["image_size"]
    dummy_input = torch.randn(1, 6, image_size, image_size)

    out_path = Path(output_path) if output_path else resolve_path(cfg["deploy"]["onnx_path"])
    ensure_dir(out_path.parent)

    # dynamo=False forces the legacy TorchScript-based exporter, which
    # doesn't require the optional `onnxscript` package. Newer torch
    # versions default to the dynamo exporter; if you have onnxscript
    # installed and prefer that path, drop this argument.
    torch.onnx.export(
        wrapped,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["tamper_prob_map"],
        dynamic_axes={"input": {0: "batch"}, "tamper_prob_map": {0: "batch"}},
        opset_version=cfg["deploy"]["opset_version"],
        dynamo=False,
    )
    print(f"[export_onnx] wrote {out_path}")

    # quick sanity check: run onnxruntime on the same dummy input and
    # compare against the PyTorch output, so a broken export fails loudly.
    import onnxruntime as ort
    import numpy as np

    with torch.no_grad():
        torch_out = wrapped(dummy_input).numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"[export_onnx] max abs diff (torch vs onnxruntime) on dummy input: {max_diff:.6f}")
    if max_diff > 1e-3:
        print("[export_onnx] WARNING: diff larger than expected — inspect the export.")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    ckpt_path = resolve_path(args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[export_onnx] checkpoint not found at {ckpt_path}. Run model/train.py first.")
        sys.exit(1)

    export(cfg, str(ckpt_path), args.output)


if __name__ == "__main__":
    main()
