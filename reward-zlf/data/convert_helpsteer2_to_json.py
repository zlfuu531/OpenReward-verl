#!/usr/bin/env python3
"""Download NVIDIA HelpSteer2 dataset from HF and convert to verl json format
   (single-file JSON with list of objects).

Hard-coded paths as requested.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import datasets  # type: ignore

# ------------------------- CONSTANTS -----------------------------
DATA_SOURCE = "helpsteer2"
ABILITY = "preference"
HF_NAME = "nvidia/HelpSteer2"
OUT_DIR = Path("/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data")
TRAIN_FILE = OUT_DIR / "helpsteer2_train.json"
TEST_FILE = OUT_DIR / "helpsteer2_test.json"
TRAIN_SIZE = 5000
TEST_SIZE = 500
RANDOM_SEED = 42
# ----------------------------------------------------------------


def map_record(ex: dict, split: str, idx: int) -> dict:
    """Convert a raw HelpSteer2 row to verl-compatible dict.

    The dataset schema has varied slightly between versions. We attempt to
    support both common variants. Expected keys:
      - "prompt" OR "conversation" OR "query"
      - two candidate responses, typically ("response_a", "response_b") **OR**
        list under "responses".
      - A preference indicator ⇒ which response is better.
    """
    # 1. prompt text
    prompt = (
        ex.get("prompt")
        or ex.get("conversation")
        or ex.get("query")
        or ""
    )

    # 2. candidate responses
    if "response_a" in ex and "response_b" in ex:
        resp_a, resp_b = ex["response_a"], ex["response_b"]
    elif "responses" in ex and isinstance(ex["responses"], list) and len(ex["responses"]) >= 2:
        resp_a, resp_b = ex["responses"][0], ex["responses"][1]
    else:
        raise KeyError("Cannot locate candidate responses in sample: keys=" + str(ex.keys()))

    # 3. preference / ground truth choice
    # common patterns: "preference" == "A"/"B"  OR "better_response_id" == 0/1
    if "preference" in ex:
        choice = "A" if str(ex["preference"]).strip().upper() in {"A", "0"} else "B"
    elif "better_response_id" in ex:
        choice = "A" if int(ex["better_response_id"]) == 0 else "B"
    else:
        # fallback: assume first is better
        choice = "A"

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ABILITY,
        "reward_model": {
            "style": "rule",
            "ground_truth_choice": choice,
        },
        "extra_info": {"split": split, "index": idx},
        "response_A": resp_a,
        "response_B": resp_b,
    }


def sample_and_convert(hf_split: str, keep_n: int, split_name: str):
    ds = datasets.load_dataset(HF_NAME, split=hf_split)
    # deterministic shuffle + sub-sample
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:keep_n]

    output = [map_record(ds[i], split_name, i) for i in indices]
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading & processing HelpSteer2 …")
    train_data = sample_and_convert("train", TRAIN_SIZE, "train")
    test_data = sample_and_convert("validation", TEST_SIZE, "test")

    # save
    with TRAIN_FILE.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with TEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("Saved:", TRAIN_FILE, "and", TEST_FILE)


if __name__ == "__main__":
    main()

