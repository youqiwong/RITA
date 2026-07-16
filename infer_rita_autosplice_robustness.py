#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from rita import RITA
from autosplice_robustness_protocol import (
    apply_corruption,
    read_lines,
    resolve_pair,
    unique_inference_conditions,
)
from test import AutoregressiveMaskGenerator


IMAGE_SIZE = (512, 512)


def extract_mask(generated_masks, eos_token_id, num_classes):
    mask_sequence = torch.stack(generated_masks, dim=0).unsqueeze(1)
    eos_ratio = (mask_sequence == eos_token_id).float().mean(dim=(1, 2))
    eos_steps = (eos_ratio >= 0.9).nonzero(as_tuple=True)[0]
    target = mask_sequence[eos_steps[0] - 1] if len(eos_steps) > 0 else mask_sequence[-1]
    values = torch.unique(target)
    values = values[values != (num_classes - 1)]
    maximum = values.max() if len(values) > 0 else None
    return torch.where(target == maximum, 0, 1) if maximum is not None else torch.ones_like(target)


def output_name(index, image_path):
    return f"{index:06d}_{Path(image_path).stem}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args()

    pairs = [resolve_pair(line, Path(args.manifest).parent) for line in read_lines(args.manifest)]
    assigned = [(index, pair) for index, pair in enumerate(pairs) if index % args.num_chunks == args.chunk_idx]
    jobs = unique_inference_conditions()
    out_root = Path(args.out_dir)
    for key, _ in jobs:
        (out_root / key).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = RITA(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    generator = AutoregressiveMaskGenerator(model, image_size=IMAGE_SIZE, max_steps=10)
    transform = T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    iterator = tqdm(assigned, desc=f"RITA robustness rank{args.chunk_idx}", dynamic_ncols=True)
    for sample_index, (image_path, _) in iterator:
        destinations = {
            key: out_root / key / output_name(sample_index, image_path)
            for key, _ in jobs
        }
        pending = [(key, condition) for key, condition in jobs if not destinations[key].exists()]
        if not pending:
            continue
        try:
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            for key, (corruption, level) in pending:
                attacked = rgb if key == "clean" else apply_corruption(
                    rgb, corruption, level, args.seed, sample_index
                )
                tensor = transform(Image.fromarray(attacked, mode="RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    generated = generator.generate(tensor)
                mask = extract_mask(generated[0], generator.eos_token_id, args.num_classes)
                array = (mask.squeeze().cpu().numpy() * 255).astype("uint8")
                Image.fromarray(array).save(destinations[key])
        except Exception as exc:
            print(f"[warn] {image_path}: {exc}", flush=True)

    print(f"[done] chunk={args.chunk_idx} samples={len(assigned)} conditions={len(jobs)}")


if __name__ == "__main__":
    main()
