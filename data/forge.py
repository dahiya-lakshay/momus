"""
Synthetic forgery generator.

Takes clean document images from data/clean/ and produces tampered
versions + binary tamper masks in data/forged/<manipulation_type>/,
using four manipulation types:

  1. copy_move          — copy a patch to another location in the SAME image
  2. splice             — paste a patch from a DIFFERENT clean image (region splicing,
                           e.g. swapping in another document's DOB/name field)
  3. font_substitution  — overwrite a known text-field region with new
                           PIL-rendered text (simulates editing a field value)
  4. inpaint_removal    — remove a region via OpenCV inpainting (simulates
                           erasing a stamp/watermark/signature)

Each output sample is saved as:
    data/forged/<type>/<stem>_fake.png   (tampered image)
    data/forged/<type>/<stem>_mask.png   (binary mask, 255 = tampered pixel)
    data/forged/<type>/<stem>_meta.json  (bookkeeping: source files, patch coords)

The `holdout_type` in config.yaml (default: inpaint_removal) is
generated but should be EXCLUDED from training data by dataset.py's
cross-manipulation split, and used only for the generalization-gap
eval in eval/cross_manipulation.py.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, set_seed, save_json  # noqa: E402
from data.synthetic_docs import _get_font, _random_date, _random_id_number, FIRST_NAMES, LAST_NAMES  # noqa: E402


def _load_clean_images(clean_dir: Path) -> list[Path]:
    return sorted(clean_dir.glob("*.png"))


def _random_patch_box(w: int, h: int, area_frac: float, rng: random.Random) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for a rectangular patch covering ~area_frac
    of the image, placed at a random valid location."""
    target_area = area_frac * w * h
    aspect = rng.uniform(0.6, 1.6)
    ph = int(np.sqrt(target_area / aspect))
    pw = int(ph * aspect)
    ph = max(8, min(ph, h - 2))
    pw = max(8, min(pw, w - 2))
    x1 = rng.randint(0, w - pw - 1)
    y1 = rng.randint(0, h - ph - 1)
    return x1, y1, x1 + pw, y1 + ph


def copy_move(img: np.ndarray, rng: random.Random, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Copy a patch to a different location within the same image."""
    h, w = img.shape[:2]
    frac = rng.uniform(cfg["copy_move"]["min_patch_frac"], cfg["copy_move"]["max_patch_frac"])
    sx1, sy1, sx2, sy2 = _random_patch_box(w, h, frac, rng)
    patch = img[sy1:sy2, sx1:sx2].copy()
    ph, pw = patch.shape[:2]

    # find a destination that doesn't heavily overlap the source
    for _ in range(20):
        dx1 = rng.randint(0, w - pw - 1)
        dy1 = rng.randint(0, h - ph - 1)
        if abs(dx1 - sx1) > pw // 2 or abs(dy1 - sy1) > ph // 2:
            break
    dx2, dy2 = dx1 + pw, dy1 + ph

    out = img.copy()
    out[dy1:dy2, dx1:dx2] = patch
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy1:dy2, dx1:dx2] = 255
    return out, mask


def splice(img: np.ndarray, donor: np.ndarray, rng: random.Random, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Paste a patch from a DIFFERENT (donor) clean image into img."""
    h, w = img.shape[:2]
    dh, dw = donor.shape[:2]
    frac = rng.uniform(cfg["splice"]["min_patch_frac"], cfg["splice"]["max_patch_frac"])
    sx1, sy1, sx2, sy2 = _random_patch_box(dw, dh, frac, rng)
    patch = donor[sy1:sy2, sx1:sx2].copy()
    ph, pw = patch.shape[:2]
    ph, pw = min(ph, h - 2), min(pw, w - 2)
    patch = cv2.resize(patch, (pw, ph))

    dx1 = rng.randint(0, w - pw - 1)
    dy1 = rng.randint(0, h - ph - 1)
    dx2, dy2 = dx1 + pw, dy1 + ph

    out = img.copy()
    out[dy1:dy2, dx1:dx2] = patch
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy1:dy2, dx1:dx2] = 255
    return out, mask


def font_substitution(img: np.ndarray, meta: dict, rng: random.Random, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite one known text-field bbox (from synthetic metadata) with
    freshly rendered replacement text, simulating a field edit (e.g. a
    changed DOB or name). If no field metadata is available, falls back
    to a random box with rendered text (still a valid 'text edit' style
    forgery, just without semantic field targeting)."""
    h, w = img.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    fields = meta.get("fields", {}) if meta else {}
    font_cfg = cfg["font_substitution"]
    font_size = max(9, int(h * font_cfg["font_size_frac"]))
    font = _get_font(font_size, font_cfg.get("font_path"))

    if fields:
        label = rng.choice(list(fields.keys()))
        bbox = fields[label]["bbox"]  # [x1, y1, x2, y2] from original render
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 3
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    else:
        bw, bh = int(w * 0.35), font_size + 6
        x1 = rng.randint(0, max(1, w - bw - 1))
        y1 = rng.randint(0, max(1, h - bh - 1))
        x2, y2 = x1 + bw, y1 + bh

    # cover the old text with a background-colored rectangle, then draw new text
    bg_sample = img[max(0, y1 - 2), max(0, x1 - 2)].tolist()
    bg_rgb = (bg_sample[2], bg_sample[1], bg_sample[0])
    draw.rectangle([x1, y1, x2, y2], fill=bg_rgb)

    if label if fields else None:
        if "DOB" in label or "EXP" in label:
            new_value = _random_date(rng)
        elif "ID" in label:
            new_value = _random_id_number(rng)
        else:
            new_value = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    else:
        new_value = _random_id_number(rng)

    draw.text((x1 + 2, y1 + 2), new_value, fill=(10, 10, 10), font=font)

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return out, mask


def inpaint_removal(img: np.ndarray, rng: random.Random, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Remove a region (e.g. a stamp/signature) via OpenCV inpainting."""
    h, w = img.shape[:2]
    frac = rng.uniform(0.03, 0.10)
    x1, y1, x2, y2 = _random_patch_box(w, h, frac, rng)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    method_cfg = cfg["inpaint_removal"]
    flag = cv2.INPAINT_TELEA if method_cfg["method"] == "telea" else cv2.INPAINT_NS
    out = cv2.inpaint(img, mask, method_cfg["inpaint_radius"], flag)
    return out, mask


MANIPULATION_FUNCS = {
    "copy_move": copy_move,
    "splice": splice,
    "font_substitution": font_substitution,
    "inpaint_removal": inpaint_removal,
}


def generate_forgeries(cfg: dict) -> dict[str, int]:
    forge_cfg = cfg["forgery"]
    clean_dir = resolve_path(cfg["data"]["clean_dir"])
    forged_root = ensure_dir(resolve_path(cfg["data"]["forged_dir"]))

    clean_paths = _load_clean_images(clean_dir)
    if len(clean_paths) < 2:
        raise RuntimeError(
            f"Need at least 2 clean images in {clean_dir} to generate forgeries "
            f"(splice needs a donor image). Run data/download.py first."
        )

    rng = random.Random(cfg["project"]["seed"])
    counts = {t: 0 for t in forge_cfg["types"]}

    for i, clean_path in enumerate(clean_paths):
        img_bgr = cv2.imread(str(clean_path))
        if img_bgr is None:
            continue
        meta_path = clean_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        for manip_type in forge_cfg["types"]:
            out_dir = ensure_dir(forged_root / manip_type)
            for v in range(forge_cfg["variants_per_image"]):
                fn = MANIPULATION_FUNCS[manip_type]
                if manip_type == "splice":
                    donor_path = clean_paths[rng.randrange(len(clean_paths))]
                    while donor_path == clean_path and len(clean_paths) > 1:
                        donor_path = clean_paths[rng.randrange(len(clean_paths))]
                    donor_bgr = cv2.imread(str(donor_path))
                    out_img, mask = fn(img_bgr, donor_bgr, rng, forge_cfg)
                elif manip_type == "font_substitution":
                    out_img, mask = fn(img_bgr, meta, rng, forge_cfg)
                else:
                    out_img, mask = fn(img_bgr, rng, forge_cfg)

                stem = f"{clean_path.stem}_v{v}"
                cv2.imwrite(str(out_dir / f"{stem}_fake.png"), out_img)
                cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
                save_json(
                    {"source": str(clean_path), "manipulation": manip_type, "variant": v},
                    out_dir / f"{stem}_meta.json",
                )
                counts[manip_type] += 1

    print("[forge] generated forgeries per type:")
    for t, c in counts.items():
        print(f"    {t:20s}: {c}")
    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    generate_forgeries(cfg)
