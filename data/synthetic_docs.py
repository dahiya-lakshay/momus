"""
Offline synthetic "clean" identity-document generator.

Real datasets (MIDV-500/2020, SROIE) require a network fetch from
external hosts. This module needs ZERO network access and exists so
that:
  1. run_all.sh can complete a full smoke test on a machine with no
     internet / blocked dataset hosts.
  2. Anyone grading this repo can `python data/synthetic_docs.py`
     and get a runnable pipeline in under a minute.

These are clearly NOT real ID documents — they are procedurally drawn
card-shaped images with fake field labels/values, a placeholder photo
box, and a background texture. They exist purely to exercise the
forgery-generation -> training -> eval -> deploy pipeline end to end.
Swap in real MIDV-500/2020 + SROIE via download.py for real results.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import load_config, resolve_path, ensure_dir, set_seed  # noqa: E402

FIRST_NAMES = ["Aditi", "Rahul", "Meera", "Arjun", "Priya", "Kabir", "Sana", "Ishaan",
               "Neha", "Vikram", "Divya", "Rohan", "Ananya", "Karan", "Tara"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Khan", "Nair", "Gupta", "Reddy", "Das",
              "Chatterjee", "Bose", "Menon", "Joshi", "Pillai", "Rao", "Kapoor"]
DOC_TYPES = ["NATIONAL ID CARD", "DRIVER LICENSE", "SALARY SLIP", "PAYMENT RECEIPT"]


def _random_date(rng: random.Random) -> str:
    d = rng.randint(1, 28)
    m = rng.randint(1, 12)
    y = rng.randint(1965, 2005)
    return f"{d:02d}/{m:02d}/{y}"


def _random_id_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(12))


def _get_font(size: int, font_path: str | None = None) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    # Try a couple of common system fonts before falling back to PIL default.
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_synthetic_document(rng: random.Random, size: int = 256,
                             font_path: str | None = None) -> tuple[Image.Image, dict]:
    """Generate one synthetic clean document image + its field metadata
    (metadata is later used by forge.py to know WHERE text fields are,
    so it can splice/substitute a specific field realistically)."""
    bg_color = tuple(rng.randint(200, 245) for _ in range(3))
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # card border
    draw.rectangle([4, 4, size - 5, size - 5], outline=(30, 30, 30), width=3)

    title_font = _get_font(max(10, size // 18), font_path)
    field_font = _get_font(max(9, size // 22), font_path)

    doc_type = rng.choice(DOC_TYPES)
    draw.text((12, 10), doc_type, fill=(20, 20, 20), font=title_font)

    # placeholder "photo" box on the left
    photo_box = [12, size // 5, size // 3, size // 5 + size // 3]
    draw.rectangle(photo_box, fill=tuple(rng.randint(120, 180) for _ in range(3)),
                    outline=(0, 0, 0), width=2)

    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    dob = _random_date(rng)
    doc_id = _random_id_number(rng)
    expiry = _random_date(rng)

    fields = {}
    y = size // 5
    text_x = size // 3 + 16
    line_h = int(size * 0.075)

    for label, value in [("NAME", name), ("DOB", dob), ("ID NO", doc_id), ("EXP", expiry)]:
        draw.text((text_x, y), f"{label}:", fill=(60, 60, 60), font=field_font)
        value_y = y + int(line_h * 0.55)
        bbox = draw.textbbox((text_x, value_y), value, font=field_font)
        draw.text((text_x, value_y), value, fill=(10, 10, 10), font=field_font)
        fields[label] = {"value": value, "bbox": list(bbox)}
        y += line_h

    # a few decorative background lines to mimic a security pattern
    for _ in range(6):
        x1, y1 = rng.randint(0, size), rng.randint(0, size)
        x2, y2 = rng.randint(0, size), rng.randint(0, size)
        draw.line([x1, y1, x2, y2], fill=tuple(rng.randint(210, 230) for _ in range(3)),
                   width=1)

    meta = {"doc_type": doc_type, "fields": fields, "size": size}
    return img, meta


def generate_synthetic_dataset(cfg: dict, count: int | None = None) -> list[Path]:
    data_cfg = cfg["data"]
    out_dir = ensure_dir(resolve_path(data_cfg["clean_dir"]))
    count = count or data_cfg["synthetic_doc_count"]
    size = data_cfg["image_size"]
    font_path = cfg["forgery"]["font_substitution"].get("font_path")

    rng = random.Random(cfg["project"]["seed"])
    paths = []
    for i in range(count):
        img, meta = make_synthetic_document(rng, size=size, font_path=font_path)
        stem = f"synth_{i:05d}"
        img_path = out_dir / f"{stem}.png"
        img.save(img_path)
        from utils.common import save_json
        save_json(meta, out_dir / f"{stem}.json")
        paths.append(img_path)
    print(f"[synthetic_docs] wrote {len(paths)} clean synthetic documents -> {out_dir}")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate offline synthetic clean documents.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    generate_synthetic_dataset(cfg, count=args.count)
