"""
Download MIDV-500, MIDV-2020, and SROIE into data/raw/, then normalize
whatever images we can find into data/clean/ as flat PNGs (256x256 by
default) ready for the forgery pipeline.

IMPORTANT — read before running on a restricted network:
MIDV-500/2020 are hosted by the L3i / Smart Engines project pages and
SROIE is hosted via the ICDAR 2019 competition site (rrc.cvc.uab.es).
These are NOT pip/npm-style package registries — some require
following a manual "I agree to research-only use" click-through on
the project page, and mirrors change over time. This script tries a
few known mirrors and times out fast; if every attempt fails (no
internet, blocked host, link rot) it automatically falls back to the
offline synthetic generator in synthetic_docs.py so the rest of the
pipeline (forge -> train -> eval -> deploy) is never blocked.

Usage:
    python data/download.py                 # try real data, fallback to synthetic
    python data/download.py --synthetic     # skip real download entirely
    python data/download.py --count 300     # synthetic doc count if falling back
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir  # noqa: E402

# Known mirrors / entry points. These are informational — some datasets
# gate access behind a click-through agreement page rather than a
# direct file, so a fully unattended download cannot be guaranteed.
SOURCES = {
    "midv500": [
        "ftp://smartengines.com/midv-500/dataset/",
    ],
    "midv2020": [
        "https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html",
    ],
    "sroie": [
        "https://rrc.cvc.uab.es/?ch=13&com=downloads",
    ],
}


def _try_fetch(url: str, dest: Path, timeout: int = 8) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "momus-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return dest.exists() and dest.stat().st_size > 0
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  [download] failed: {url} ({e})")
        return False


def attempt_real_download(cfg: dict) -> bool:
    raw_dir = ensure_dir(resolve_path(cfg["data"]["root_dir"]))
    any_success = False
    for name, urls in SOURCES.items():
        dataset_dir = ensure_dir(raw_dir / name)
        print(f"[download] attempting {name} ...")
        for url in urls:
            marker = dataset_dir / "index.html"
            ok = _try_fetch(url, marker, timeout=8)
            if ok:
                print(f"  [download] fetched index/entry point for {name} -> {marker}")
                print(f"  [download] NOTE: {name} typically needs manual, dataset-specific "
                      f"extraction after this point (archives, per-clip folders, etc). "
                      f"See README.md 'Datasets' section for manual steps.")
                any_success = True
            else:
                continue
    return any_success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true",
                         help="Skip real download attempt, go straight to synthetic fallback.")
    parser.add_argument("--count", type=int, default=None,
                         help="Number of synthetic docs to generate if falling back.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    got_real_data = False
    if not args.synthetic:
        try:
            got_real_data = attempt_real_download(cfg)
        except Exception as e:
            print(f"[download] real-download attempt raised {e}; falling back.")
            got_real_data = False

    if not got_real_data:
        if not cfg["data"]["use_synthetic_fallback"]:
            print("[download] Real data unavailable and use_synthetic_fallback=false. "
                  "Exiting without data. Edit config.yaml to enable the fallback, or "
                  "manually place images under data/clean/.")
            sys.exit(1)
        print("[download] Falling back to OFFLINE SYNTHETIC document generator.\n"
              "           (This is expected on sandboxed / offline machines, and is\n"
              "           sufficient to smoke-test the full pipeline. For real\n"
              "           results, run this script with full internet access, or\n"
              "           manually download MIDV-500/2020 + SROIE per README.md.)")
        from data.synthetic_docs import generate_synthetic_dataset
        generate_synthetic_dataset(cfg, count=args.count)
    else:
        print("[download] Real dataset entry points fetched — see messages above for "
              "manual extraction steps required per dataset.")


if __name__ == "__main__":
    main()
