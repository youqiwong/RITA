#!/usr/bin/env python3
"""Evaluate a RITA checkpoint on image,mask txt manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from rita import RITA
from test import AutoregressiveMaskGenerator


def read_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = next(csv.reader([line])) if "," in line else line.split(None, 1)
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_no}: expected image,mask")
            pairs.append((fields[0].strip(), fields[1].strip()))
    if not pairs:
        raise ValueError(f"empty manifest: {path}")
    return pairs


class TxtDataset(Dataset):
    def __init__(self, pairs, image_size):
        self.pairs = pairs
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Resize(
            (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = self.image_transform(Image.open(image_path).convert("RGB"))
        mask = self.mask_transform(Image.open(mask_path).convert("L"))
        return image, torch.from_numpy((np.array(mask) > 127).astype(np.float32))


def parse_manifest(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"manifest must be LABEL:PATH, got {spec}")
    label, path = spec.split(":", 1)
    if not label or not Path(path).is_file():
        raise FileNotFoundError(f"invalid manifest: {spec}")
    return label, path


def scores_from_sequence(sequence, eos_token_id, padding_id):
    stack = torch.stack(sequence, dim=0)
    eos_ratio = (stack == eos_token_id).float().mean(dim=(1, 2))
    eos_indices = (eos_ratio >= 0.9).nonzero(as_tuple=True)[0]
    if len(eos_indices) and eos_indices[0].item() > 0:
        target = stack[eos_indices[0].item() - 1]
    else:
        target = stack[-1]
    values = torch.unique(target)
    values = values[values != padding_id]
    max_value = values.max() if len(values) else None
    pred = torch.ones_like(target, dtype=torch.bool)
    if max_value is not None:
        pred = target != max_value
    return pred


def evaluate_one(label, path, model, device, image_size, batch_size, rank, world_size):
    dataset = TxtDataset(read_pairs(path), image_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=False)
    generator = AutoregressiveMaskGenerator(model, image_size=(image_size, image_size), max_steps=10)
    f1_sum = torch.zeros(1, device=device, dtype=torch.float64)
    iou_sum = torch.zeros(1, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.float64)
    for images, masks in tqdm(loader, desc=label, disable=rank != 0):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).bool()
        with torch.no_grad():
            sequences = generator.generate(images)
        for i, sequence in enumerate(sequences):
            pred = scores_from_sequence(sequence, generator.eos_token_id, generator.padding_id)
            gt = masks[i]
            tp = (pred & gt).sum().double()
            fp = (pred & ~gt).sum().double()
            fn = ((~pred) & gt).sum().double()
            f1_sum += 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou_sum += tp / (tp + fp + fn + 1e-8)
            count += 1
    for tensor in (f1_sum, iou_sum, count):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {"label": label, "n": int(count.item()),
            "f1": float((f1_sum / count).item()),
            "iou": float((iou_sum / count).item())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-txts", nargs="+", required=True,
                        help="LABEL:/absolute/path/to/pairs.txt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = RITA(num_classes=4).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    results = []
    for spec in args.test_txts:
        label, path = parse_manifest(spec)
        results.append(evaluate_one(label, path, model, device, args.image_size,
                                    args.batch_size, rank, world_size))
    if rank == 0:
        grouped = defaultdict(list)
        for result in results:
            key = "PIXAR" if result["label"].startswith("PIXAR") else result["label"]
            grouped[key].append(result)
        final = []
        for key, items in grouped.items():
            final.append({"label": key, "n": sum(x["n"] for x in items),
                          "f1": sum(x["f1"] for x in items) / len(items),
                          "iou": sum(x["iou"] for x in items) / len(items)})
        print("\nDataset\tF1\tIoU\tN")
        for item in final:
            print(f"{item['label']}\t{item['f1'] * 100:.2f}\t{item['iou'] * 100:.2f}\t{item['n']}")
        ood = [x for x in final if x["label"] not in {"MagicBrush", "DiffSeg30k"}]
        if ood:
            print(f"Avg OOD\t{sum(x['f1'] for x in ood) / len(ood) * 100:.2f}\t"
                  f"{sum(x['iou'] for x in ood) / len(ood) * 100:.2f}\t{len(ood)} groups")
        if args.output:
            Path(args.output).write_text(json.dumps(final, indent=2), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
