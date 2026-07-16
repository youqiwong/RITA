#!/usr/bin/env python3
import argparse

from sidset_robustness_protocol import prepare_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    path = prepare_manifest(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        workers=args.workers,
    )
    print(path)


if __name__ == "__main__":
    main()
