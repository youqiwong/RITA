"""Fixed AutoSplice robustness protocol shared by RITA robustness scripts."""

from __future__ import annotations

import csv
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


AUTOSPLICE_TXT = Path("/pubdata/wangyq/Projects/Datasets/AIGC-Loc-Testsets/AutoSplice_tp/AutoSplice_tp.txt")
DEFAULT_SAMPLE_COUNT = 3621
CORRUPTION_LEVELS = {
    "noise": [0, 2, 4, 6, 8, 10, 12],
    "blur": [0, 2, 4, 6, 8, 10, 12],
    "jpeg": [100, 95, 90, 85, 80, 75, 70],
    "resize": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
}


def level_text(level):
    return f"{float(level):g}"


def condition_key(corruption, level):
    if (corruption == "noise" and float(level) == 0) or (
        corruption == "blur" and float(level) == 0
    ) or (corruption == "resize" and float(level) == 1):
        return "clean"
    return f"{corruption}_{level_text(level)}"


def unique_inference_conditions():
    jobs = {}
    for corruption, levels in CORRUPTION_LEVELS.items():
        for level in levels:
            jobs.setdefault(condition_key(corruption, level), (corruption, level))
    return list(jobs.items())


def apply_corruption(image, corruption, level, seed, sample_index):
    """Apply deterministic degradation to an RGB uint8 image."""
    if corruption == "noise":
        sigma = float(level)
        if sigma == 0:
            return image.copy()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(sample_index), 1729]))
        noise = rng.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if corruption == "blur":
        sigma = float(level)
        if sigma == 0:
            return image.copy()
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if corruption == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=int(level))
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"))
    if corruption == "resize":
        scale = float(level)
        if scale == 1:
            return image.copy()
        if not 0 < scale <= 1:
            raise ValueError(f"Resize scale must be in (0, 1]: {scale}")
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown corruption: {corruption}")


def stratified_sample_indices(ratios, sample_count=DEFAULT_SAMPLE_COUNT, seed=42, strata=10):
    ratios = np.asarray(ratios, dtype=np.float64)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count >= len(ratios):
        return list(range(len(ratios)))
    strata = max(1, min(strata, sample_count, len(ratios)))
    groups = np.array_split(np.argsort(ratios, kind="stable"), strata)
    base, extra = divmod(sample_count, strata)
    rng = np.random.default_rng(seed)
    selected = []
    for group_index, group in enumerate(groups):
        take = base + int(group_index < extra)
        selected.extend(int(index) for index in rng.choice(group, take, replace=False))
    return sorted(selected)


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_pair(line, root):
    parts = [part.strip() for part in line.rsplit(",", 1)]
    if len(parts) != 2:
        raise ValueError(f"Expected image,mask pair: {line}")
    paths = []
    for value in parts:
        path = Path(value)
        paths.append((path if path.is_absolute() else Path(root) / path).resolve())
    return tuple(paths)


def _mask_ratio(pair):
    mask = cv2.imread(str(pair[1]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(pair[1])
    return float(np.count_nonzero(mask > 127) / mask.size)


def prepare_manifest(source_txt=AUTOSPLICE_TXT, output_dir="robustness_results/autosplice_all_seed42/manifests",
                     sample_count=DEFAULT_SAMPLE_COUNT, seed=42, workers=32):
    source_txt = Path(source_txt).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"autosplice_n{sample_count}_seed{seed}"
    manifest = output_dir / f"{stem}.txt"
    metadata_path = output_dir / f"{stem}.json"
    expected = {"source_txt": str(source_txt), "sample_count": sample_count, "seed": seed, "strata": 10}
    if manifest.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            if len(read_lines(manifest)) == metadata.get("selected_count"):
                return manifest

    pairs = [resolve_pair(line, source_txt.parent) for line in read_lines(source_txt)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        ratios = list(pool.map(_mask_ratio, pairs, chunksize=32))
    indices = stratified_sample_indices(ratios, sample_count, seed)
    manifest.write_text(
        "".join(f"{pairs[index][0]},{pairs[index][1]}\n" for index in indices),
        encoding="utf-8",
    )
    selected_ratios = [ratios[index] for index in indices]
    metadata = {
        **expected,
        "source_count": len(pairs),
        "selected_count": len(indices),
        "selected_indices": indices,
        "foreground_ratio_min": min(selected_ratios),
        "foreground_ratio_mean": float(np.mean(selected_ratios)),
        "foreground_ratio_max": max(selected_ratios),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest


def cleanup_prediction_tree(pred_root, keep=False):
    """Remove temporary predictions only after metrics were written successfully."""
    pred_root = Path(pred_root)
    if not keep and pred_root.exists():
        shutil.rmtree(pred_root)


def write_results(output_dir, method, manifest, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["Method", "Corruption", "Level", "Samples", "Missing", "Pixel_F1", "IoU", "Seed", "Manifest"]
    long_path = output_dir / "autosplice_robustness_long.tsv"
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Method": method,
                "Corruption": row["corruption"],
                "Level": level_text(row["level"]),
                "Samples": row["samples"],
                "Missing": row["missing"],
                "Pixel_F1": f"{row['f1']:.6f}",
                "IoU": f"{row['iou']:.6f}",
                "Seed": 42,
                "Manifest": str(Path(manifest).resolve()),
            })
    for metric, filename in (("f1", "autosplice_robustness_f1.tsv"), ("iou", "autosplice_robustness_iou.tsv")):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["Method"] + [
                f"{kind}_{level_text(level)}" for kind, levels in CORRUPTION_LEVELS.items() for level in levels
            ])
            writer.writerow([method] + [f"{row[metric]:.6f}" for row in rows])
    return long_path
