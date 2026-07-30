"""
Post-training int8 quantization of the exported ONNX model, using
onnxruntime's dynamic quantization (weights quantized to int8,
activations quantized on the fly at inference — no calibration
dataset required, which keeps this a true zero-budget step).

Usage:
    python deploy/quantize.py
    (Run deploy/export_onnx.py first to produce the fp32 .onnx file.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir  # noqa: E402


def quantize(cfg: dict, input_path: str = None, output_path: str = None):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    in_path = Path(input_path) if input_path else resolve_path(cfg["deploy"]["onnx_path"])
    out_path = Path(output_path) if output_path else resolve_path(cfg["deploy"]["onnx_int8_path"])
    ensure_dir(out_path.parent)

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found — run deploy/export_onnx.py first.")

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )

    fp32_size = in_path.stat().st_size / 1e6
    int8_size = out_path.stat().st_size / 1e6
    print(f"[quantize] fp32 model: {fp32_size:.2f} MB  ->  int8 model: {int8_size:.2f} MB "
          f"({(1 - int8_size / fp32_size) * 100:.1f}% smaller)")
    print(f"[quantize] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    quantize(cfg, args.input, args.output)


if __name__ == "__main__":
    main()
