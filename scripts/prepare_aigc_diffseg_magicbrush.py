#!/usr/bin/env python3
"""Convert AIGC-Localization txt pair lists to the JSON format used by RITA.

Each input line is expected to contain an image path and a mask path separated
by a comma, tab, or whitespace.  Blank/comment lines are ignored.  DiffSeg is
truncated *before* any optional split, preserving the requested first-10k
source-order protocol; MagicBrush is kept in full by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


def _parse_pair(line: str, source: Path, line_no: int) -> tuple[str, str]:
    text = line.strip()
    if not text or text.startswith("#"):
        raise ValueError("skip")

    # The released manifests use comma-separated pairs, but accepting tab and
    # whitespace makes the converter robust to the common txt variants.
    if "," in text:
        fields = next(csv.reader([text], skipinitialspace=True))
    elif "\t" in text:
        fields = text.split("\t")
    else:
        fields = re.split(r"\s+", text, maxsplit=1)
    fields = [field.strip() for field in fields if field.strip()]
    if len(fields) < 2:
        raise ValueError(f"{source}:{line_no}: expected image and mask paths")
    return fields[0], fields[1]


def _normalize_path(value: str, manifest: Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    # Preserve an already-valid path relative to the caller's working
    # directory; otherwise resolve the common manifest-relative convention.
    if candidate.is_file():
        return str(candidate)
    return str((manifest.parent / candidate).resolve())


def read_pairs(path: Path, limit: int | None = None) -> list[list[str]]:
    pairs: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                image, mask = _parse_pair(line, path, line_no)
            except ValueError as exc:
                if str(exc) == "skip":
                    continue
                raise
            pairs.append([_normalize_path(image, path), _normalize_path(mask, path)])
            if limit is not None and len(pairs) >= limit:
                break
    if not pairs:
        raise RuntimeError(f"No valid image/mask pairs found in {path}")
    return pairs


def _assert_files(pairs: Iterable[list[str]], label: str) -> None:
    missing = [(image, mask) for image, mask in pairs
               if not Path(image).is_file() or not Path(mask).is_file()]
    if missing:
        sample = missing[:3]
        raise FileNotFoundError(
            f"{label}: {len(missing)} pair(s) have missing files; examples={sample}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffseg-txt", required=True, type=Path)
    parser.add_argument("--magicbrush-txt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diffseg-limit", type=int, default=10000)
    parser.add_argument("--magicbrush-limit", type=int, default=0,
                        help="0 keeps the complete MagicBrush manifest")
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    if args.diffseg_limit <= 0:
        raise ValueError("--diffseg-limit must be positive")
    mb_limit = args.magicbrush_limit or None
    diffseg = read_pairs(args.diffseg_txt, args.diffseg_limit)
    magicbrush = read_pairs(args.magicbrush_txt, mb_limit)
    if args.check_files:
        _assert_files(diffseg, "DiffSeg30k")
        _assert_files(magicbrush, "MagicBrush")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diffseg_json = args.output_dir / "DiffSeg30k_first10000.json"
    magicbrush_json = args.output_dir / "MagicBrush_full.json"
    config_json = args.output_dir / "rita_aigc_config.json"
    for path, pairs in ((diffseg_json, diffseg), (magicbrush_json, magicbrush)):
        path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    config = [["JsonDataset", str(diffseg_json)],
              ["JsonDataset", str(magicbrush_json)]]
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"[prepare] DiffSeg30k first-order pairs: {len(diffseg)}")
    print(f"[prepare] MagicBrush pairs: {len(magicbrush)}")
    print(f"[prepare] config: {config_json}")


if __name__ == "__main__":
    main()
