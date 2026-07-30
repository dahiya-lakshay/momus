"""
Benchmark fp32 vs int8 ONNX models: CPU inference latency (ms/image)
and accuracy delta (pixel IoU/F1) on real held-out samples.

Usage:
    python deploy/benchmark.py --checkpoint checkpoints/best.pt
    (Runs export_onnx.py + quantize.py first if artifacts are missing.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, save_json, markdown_table, load_json  # noqa: E402
from data.dataset import ForgeryDataset  # noqa: E402
from data.srm import compute_srm_residual  # noqa: E402
from eval.metrics import batch_pixel_metrics  # noqa: E402


def _prepare_inputs(samples: list[dict], image_size: int, n: int):
    """Load up to n samples as raw 6-channel [rgb||srm] numpy arrays
    (NCHW, float32) plus their ground-truth masks, without going through
    the PyTorch DataLoader (keeps this script torch-independent so it
    reflects a pure ONNX Runtime deployment)."""
    import cv2

    imgs, gts = [], []
    for s in samples[:n]:
        img = cv2.imread(s["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size))
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        rgb = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
        srm = compute_srm_residual(img)

        combined = np.concatenate([rgb, srm], axis=-1)  # H,W,6
        combined = np.transpose(combined, (2, 0, 1))     # 6,H,W
        imgs.append(combined)
        gts.append((mask > 127).astype(np.float32))

    return np.stack(imgs, axis=0).astype(np.float32), np.stack(gts, axis=0)


def _run_onnx(session, inputs: np.ndarray, warmup: int = 3) -> tuple[np.ndarray, float]:
    """Run inference one image at a time (batch=1) to get realistic
    single-image CPU latency, returns (all_probs, avg_ms_per_image).

    A few warmup iterations are run first and excluded from timing —
    onnxruntime does lazy kernel/session initialization on first run,
    and without a warmup pass that one-time cost gets misattributed to
    "inference latency", which is especially misleading for int8
    (its op-level dispatch overhead on first call is larger than fp32's).
    """
    n = inputs.shape[0]
    warmup = min(warmup, n)
    for i in range(warmup):
        session.run(None, {"input": inputs[i:i + 1]})

    all_probs = []
    times = []
    for i in range(n):
        single = inputs[i:i + 1]
        t0 = time.perf_counter()
        out = session.run(None, {"input": single})[0]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        all_probs.append(out[0, 0])
    return np.stack(all_probs, axis=0), float(np.mean(times))


def benchmark(cfg: dict, fp32_path: str = None, int8_path: str = None):
    fp32_path = Path(fp32_path) if fp32_path else resolve_path(cfg["deploy"]["onnx_path"])
    int8_path = Path(int8_path) if int8_path else resolve_path(cfg["deploy"]["onnx_int8_path"])

    for p in [fp32_path, int8_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run deploy/export_onnx.py and deploy/quantize.py first."
            )

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    test_samples = load_json(splits_dir / "test.json")
    n = min(cfg["deploy"]["benchmark_num_images"], len(test_samples))
    if n == 0:
        print("[benchmark] no test samples available — skipping.")
        return None

    inputs, gts = _prepare_inputs(test_samples, cfg["data"]["image_size"], n)
    print(f"[benchmark] running on {n} images (CPUExecutionProvider)")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # single-thread for a fair, reproducible ms/image number
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    fp32_sess = ort.InferenceSession(str(fp32_path), sess_options, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_path), sess_options, providers=["CPUExecutionProvider"])

    fp32_probs, fp32_ms = _run_onnx(fp32_sess, inputs)
    int8_probs, int8_ms = _run_onnx(int8_sess, inputs)

    if int8_ms > fp32_ms:
        print(f"[benchmark] NOTE: int8 ({int8_ms:.2f} ms) is slower than fp32 ({fp32_ms:.2f} ms) "
              f"on this machine/batch size. This is a known and documented onnxruntime behavior: "
              f"dynamic quantization inserts per-op quantize/dequantize casts whose overhead can "
              f"exceed the matmul savings on (a) very small models, (b) tiny batch=1 CPU inference, "
              f"or (c) CPUs without VNNI/AVX512-VNNI int8 acceleration. Static quantization with a "
              f"calibration set, or a CPU with proper int8 SIMD support, typically reverses this. "
              f"Report this honestly in the README rather than cherry-picking a favorable run.")

    threshold = cfg["eval"]["threshold"]
    fp32_metrics = batch_pixel_metrics(fp32_probs, gts, threshold=threshold)
    int8_metrics = batch_pixel_metrics(int8_probs, gts, threshold=threshold)

    result = {
        "n_images": n,
        "fp32": {"latency_ms": fp32_ms, **fp32_metrics,
                 "model_size_mb": fp32_path.stat().st_size / 1e6},
        "int8": {"latency_ms": int8_ms, **int8_metrics,
                 "model_size_mb": int8_path.stat().st_size / 1e6},
        "speedup_x": fp32_ms / int8_ms if int8_ms > 0 else float("nan"),
        "iou_delta": fp32_metrics["iou"] - int8_metrics["iou"],
    }

    print(f"[benchmark] fp32: {fp32_ms:.2f} ms/img, IoU={fp32_metrics['iou']:.4f}, "
          f"{result['fp32']['model_size_mb']:.1f} MB")
    print(f"[benchmark] int8: {int8_ms:.2f} ms/img, IoU={int8_metrics['iou']:.4f}, "
          f"{result['int8']['model_size_mb']:.1f} MB")
    print(f"[benchmark] speedup: {result['speedup_x']:.2f}x   IoU delta: {result['iou_delta']:+.4f}")

    results_dir = ensure_dir(resolve_path(cfg["eval"]["results_dir"]))
    save_json(result, results_dir / "deploy_benchmark.json")

    table = markdown_table(
        headers=["Precision", "Latency (ms/img, CPU, 1 thread)", "Pixel IoU", "Pixel F1", "Model size (MB)"],
        rows=[
            ["fp32", f"{fp32_ms:.2f}", f"{fp32_metrics['iou']:.4f}", f"{fp32_metrics['f1']:.4f}",
             f"{result['fp32']['model_size_mb']:.1f}"],
            ["int8", f"{int8_ms:.2f}", f"{int8_metrics['iou']:.4f}", f"{int8_metrics['f1']:.4f}",
             f"{result['int8']['model_size_mb']:.1f}"],
        ],
    )
    md = (f"# Deployment Benchmark (fp32 vs int8, CPU, n={n} images)\n\n{table}\n\n"
          f"**Speedup:** {result['speedup_x']:.2f}x  |  **IoU delta (fp32 - int8):** "
          f"{result['iou_delta']:+.4f}\n")
    if result["speedup_x"] < 1.0:
        md += (
            "\n> **Note:** int8 was slower than fp32 on this run. This is a known "
            "onnxruntime behavior on machines without AVX512-VNNI int8 acceleration, or with "
            "dynamic (rather than static/calibrated) quantization on tiny batch sizes — the "
            "quantize/dequantize overhead per op can exceed the matmul savings. Re-run "
            "`deploy/benchmark.py` on the actual target CPU before making a production "
            "int8-vs-fp32 decision, and consider static quantization with a calibration set "
            "if int8 remains slower there too.\n"
        )
    (results_dir / "deploy_benchmark.md").write_text(md)
    print(f"[benchmark] wrote {results_dir / 'deploy_benchmark.md'}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--fp32", default=None)
    parser.add_argument("--int8", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    benchmark(cfg, args.fp32, args.int8)


if __name__ == "__main__":
    main()
