#!/usr/bin/env python3
"""Convert HelpSteer3_CLI preference dataset into verl post-training parquet format.

Usage
-----
python convert_to_parquet.py \
    --data_dir zlf/guotai-reward/verl/reward-zlf/data/HelpSteer3_CLI/preference \
    --out_dir  zlf/guotai-reward/verl/reward-zlf/data/HelpSteer3_CLI/parquet \
    --train_size 5000 \
    --test_size 500

This script
1. Reads `train.jsonl.gz` and `validation.jsonl.gz` under ``data_dir``.
2. Maps each raw record to verl expected fields.
3. Sub-samples ``train_size`` and ``test_size`` examples deterministically.
4. Saves two parquet files: ``train.parquet`` and ``test.parquet``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Iterator, List, Dict, Any

import datasets  # type: ignore

DATA_SOURCE_NAME = "helpsteer3_cli"
ABILITY = "preference"


def load_jsonl_gz(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts from a .jsonl.gz file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_map_fn(split: str):
    """Return a mapper that converts raw example to verl format."""

    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Raw schema assumption
        # {
        #   "conversation": "...",  # user prompt
        #   "response_a": "...",
        #   "response_b": "...",
        #   "preference": "A" | "B"
        # }
        prompt_txt = example["conversation"]
        resp_a = example.get("response_a", "")
        resp_b = example.get("response_b", "")
        gt_choice = example.get("preference", "A")  # default A if missing

        return {
            "data_source": DATA_SOURCE_NAME,
            "prompt": [{"role": "user", "content": prompt_txt}],
            "ability": ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth_choice": gt_choice,
            },
            "extra_info": {
                "split": split,
                "index": idx,
            },
            # fields consumed by prompt template
            "response_A": resp_a,
            "response_B": resp_b,
        }

    return process_fn


def convert_split(raw_path: Path, split_name: str, max_items: int | None) -> datasets.Dataset:
    """Convert one split and optionally truncate to max_items."""
    raw_examples = list(load_jsonl_gz(raw_path))

    if max_items is not None:
        # deterministic sub-sample: use first N after a fixed shuffle for randomness
        import random

        rng = random.Random(42)
        rng.shuffle(raw_examples)
        raw_examples = raw_examples[: max_items]

    processed: List[Dict[str, Any]] = [make_map_fn(split_name)(ex, i) for i, ex in enumerate(raw_examples)]
    return datasets.Dataset.from_list(processed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="zlf/guotai-reward/verl/reward-zlf/data/HelpSteer3_CLI/preference",
        help="Directory containing train.jsonl.gz and validation.jsonl.gz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="zlf/guotai-reward/verl/reward-zlf/data/HelpSteer3_CLI/parquet",
        help="Where to save parquet files",
    )
    parser.add_argument("--train_size", type=int, default=5000, help="Samples to keep in train set")
    parser.add_argument("--test_size", type=int, default=500, help="Samples to keep in test set")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = data_dir / "train.jsonl.gz"
    val_raw = data_dir / "validation.jsonl.gz"

    if not train_raw.exists() or not val_raw.exists():
        raise FileNotFoundError("Expect train.jsonl.gz and validation.jsonl.gz in data_dir")

    print("Converting train split …")
    train_ds = convert_split(train_raw, "train", args.train_size)
    print(f"Train split size: {len(train_ds)}")

    print("Converting test split …")
    test_ds = convert_split(val_raw, "test", args.test_size)
    print(f"Test split size: {len(test_ds)}")

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"

    train_ds.to_parquet(str(train_path))
    test_ds.to_parquet(str(test_path))

    print("Saved parquet files to", out_dir)


if __name__ == "__main__":
    main()

