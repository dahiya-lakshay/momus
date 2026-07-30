#!/usr/bin/env bash
# =============================================================
# MOMUS — full pipeline, top to bottom.
#
# Usage:
#   ./run_all.sh              # smoke test (tiny data, 2 epochs) — sanity check only
#   ./run_all.sh --full       # full run (real epoch count from config.yaml)
#
# Designed to run start-to-finish on a fresh Google Colab T4 runtime,
# or on a CPU-only machine for the smoke test (slower, but works).
# =============================================================
set -euo pipefail

MODE="smoke"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
fi

echo "=============================================="
echo " MOMUS pipeline — mode: ${MODE}"
echo "=============================================="

echo ""
echo "[1/8] Installing dependencies..."
echo "      (torch/torchvision are NOT force-installed — Colab ships working ones;"
echo "       see requirements.txt if you need to install them from scratch)"
pip install -q -r requirements.txt --break-system-packages 2>/dev/null \
  || pip install -q -r requirements.txt \
  || { echo "pip install failed — try: pip install -r requirements.txt --break-system-packages"; exit 1; }

echo ""
echo "[2/8] Downloading data (falls back to offline synthetic docs if no internet)..."
if [[ "$MODE" == "smoke" ]]; then
  python data/download.py --count 16
else
  python data/download.py
fi

echo ""
echo "[3/8] Generating forgeries (copy-move, splice, font-substitution, inpaint-removal)..."
python data/forge.py

echo ""
echo "[4/8] Building train/val/test/holdout splits..."
python data/dataset.py

echo ""
echo "[5/8] Training model..."
if [[ "$MODE" == "smoke" ]]; then
  python model/train.py --smoke-test
else
  python model/train.py
fi

echo ""
echo "[6/8] Evaluating (test set, cross-manipulation generalization, robustness sweep, failure cases)..."
python eval/evaluate.py --checkpoint checkpoints/best.pt
python eval/cross_manipulation.py --checkpoint checkpoints/best.pt
python eval/robustness.py --checkpoint checkpoints/best.pt
python eval/failure_analysis.py --checkpoint checkpoints/best.pt

echo ""
echo "[7/8] Exporting to ONNX + int8 quantizing..."
python deploy/export_onnx.py --checkpoint checkpoints/best.pt
python deploy/quantize.py

echo ""
echo "[8/8] Benchmarking fp32 vs int8..."
python deploy/benchmark.py

echo ""
echo "=============================================="
echo " Done. Results in results/, checkpoints in checkpoints/,"
echo " deployable ONNX models in deploy/artifacts/."
echo ""
echo " NOTE: In '${MODE}' mode with synthetic/small data, metrics"
echo " (IoU/F1/AUC) will be near-zero or noisy — this is EXPECTED."
echo " Re-run with --full and real MIDV-500/2020 + SROIE data"
echo " (see README.md > Datasets) for meaningful numbers."
echo "=============================================="
