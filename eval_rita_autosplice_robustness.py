#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from infer_rita_autosplice_robustness import output_name
from autosplice_robustness_protocol import (
    CORRUPTION_LEVELS,
    cleanup_prediction_tree,
    condition_key,
    read_lines,
    resolve_pair,
    write_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="RITA-Retrain")
    parser.add_argument("--keep-pred-masks", action="store_true")
    args = parser.parse_args()

    pairs = [resolve_pair(line, Path(args.manifest).parent) for line in read_lines(args.manifest)]
    rows = []
    for corruption, levels in CORRUPTION_LEVELS.items():
        for level in levels:
            key = condition_key(corruption, level)
            f1_values, iou_values = [], []
            missing = 0
            iterator = tqdm(pairs, desc=f"{corruption}={level}", leave=False, dynamic_ncols=True)
            for index, (image_path, mask_path) in enumerate(iterator):
                gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(
                    str(Path(args.pred_root) / key / output_name(index, image_path)),
                    cv2.IMREAD_GRAYSCALE,
                )
                if gt is None:
                    raise FileNotFoundError(mask_path)
                if pred is None:
                    missing += 1
                    f1_values.append(0.0)
                    iou_values.append(0.0)
                    continue
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                pred_bin = pred > 127
                gt_bin = gt > 127
                intersection = np.logical_and(pred_bin, gt_bin).sum()
                pred_sum, gt_sum = pred_bin.sum(), gt_bin.sum()
                f1_values.append(float(2 * intersection / (pred_sum + gt_sum + 1e-8)))
                iou_values.append(float(intersection / (pred_sum + gt_sum - intersection + 1e-8)))
            rows.append({
                "corruption": corruption,
                "level": level,
                "samples": len(pairs),
                "missing": missing,
                "f1": float(np.mean(f1_values)),
                "iou": float(np.mean(iou_values)),
            })
            print(
                f"[done] {corruption}={level} F1={rows[-1]['f1']:.4f} "
                f"IoU={rows[-1]['iou']:.4f} missing={missing}"
            )
    path = write_results(args.output_dir, args.method, args.manifest, rows)
    print(f"TSV: {path}")
    cleanup_prediction_tree(args.pred_root, keep=args.keep_pred_masks)
    if not args.keep_pred_masks:
        print(f"Removed temporary predictions: {args.pred_root}")


if __name__ == "__main__":
    main()
