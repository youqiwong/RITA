#!/usr/bin/env python3
"""Report missing files and image/mask shape mismatches in RITA JSON pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    missing = []
    mismatched = []
    total = 0
    for kind, json_path in config:
        if kind != "JsonDataset":
            continue
        pairs = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for image_path, mask_path in pairs:
            total += 1
            image = Path(image_path)
            mask = Path(mask_path)
            if not image.is_file() or not mask.is_file():
                missing.append((str(image), str(mask)))
                continue
            with Image.open(image) as im, Image.open(mask) as gt:
                if im.size != gt.size:
                    mismatched.append((str(image), im.size, str(mask), gt.size))
    print(f"[audit] pairs={total} missing={len(missing)} shape_mismatch={len(mismatched)}")
    for item in missing[:args.max_examples]:
        print(f"[audit][missing] image={item[0]} mask={item[1]}")
    for image, image_size, mask, mask_size in mismatched[:args.max_examples]:
        print(f"[audit][shape] image={image} size={image_size} mask={mask} size={mask_size}")


if __name__ == "__main__":
    main()
