#!/usr/bin/env python3
"""Convert HelpSteer3_CLI preference dataset to verl json format.
python /nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data/convert_to_json.py   --data_dir /nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data/HelpSteer3_CLI/preference   --out_dir /nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data
Creates two files under the output directory (default project root data folder):
    train.json   (5,000 samples)
    test.json    (500 samples)
Each file is a single JSON array (list of objects), compatible with verl post-training.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Iterator, List

DATA_SOURCE_NAME = "helpsteer3_cli"
ABILITY = "preference"
RNG_SEED = 42  # deterministic subsample


def load_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts from a .jsonl.gz file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_choice(x: Any) -> str:
    if x is None:
        return "A"
    s = str(x).strip().upper()
    if s in {"A", "0", "1", "2", "RESP1", "RESPONSE1"}:
        return "A"
    if s in {"B", "3", "4", "5", "RESP2", "RESPONSE2"}:
        return "B"
    # common patterns: "response1" / "response2"
    if "1" in s:
        return "A"
    if "2" in s:
        return "B"
    return "A"


def _build_prompt_from_context(context: Any) -> str:
    """HelpSteer3_CLI uses `context` as a list of chat messages."""
    if not context:
        return ""
    if isinstance(context, list):
        parts: List[str] = []
        for msg in context:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip() or "user"
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            # Keep it simple; preserve turns.
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    return str(context)


JUDGE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below.

[Client Question]
{conv_his}

[The Start of Chatbot A's Response]
{response_A}
[The End of Chatbot A's Response]

[The Start of Chatbot B's Response]
{response_B}
[The End of Chatbot B's Response]

Output your final verdict by strictly following this format:

<critics>
[Provide a brief summary of your reasoning for the choice]
</critics>
<choice>
[[A]]
</choice>
Note: Use [[A]] if A is better, or [[B]] if B is better.
"""


def map_example(example: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """Map raw HelpSteer3_CLI record to verl expected schema.

    Observed raw schema (preference):
      - context: list[{role, content}]
      - response1 / response2
      - overall_preference: e.g. "response1" or "response2" (sometimes)
      - individual_preference: optional
    """
    # build conv history
    context = example.get("context")
    conv_his = _build_prompt_from_context(context)

    # responses
    resp_a = str(example.get("response1") or "")
    resp_b = str(example.get("response2") or "")

    # judge prompt (single-turn user message)
    prompt_text = JUDGE_TEMPLATE.format(conv_his=conv_his, response_A=resp_a, response_B=resp_b)

    # preference
    pref = example.get("overall_preference")
    if pref is None:
        pref = example.get("individual_preference")
    gt_choice = _normalize_choice(pref)

    # human critique / reasoning (process feedback)
    human_critique = ""
    ip = example.get("individual_preference")
    if isinstance(ip, list) and ip:
        first = ip[0]
        if isinstance(first, dict):
            human_critique = str(first.get("reasoning") or "")

    return {
        "data_source": DATA_SOURCE_NAME,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": ABILITY,
        # ground truth at top-level for RewardManager convenience
        "ground_truth_choice": gt_choice,
        "reward_model": {
            "style": "rule",
            # naive reward manager 期望该字段：data_item.non_tensor_batch["reward_model"]["ground_truth"]
            # 这里直接复用 choice，保证训练可跑通（避免 KeyError: 'ground_truth'）
            "ground_truth": gt_choice,
            "ground_truth_choice": gt_choice,
        },
        "extra_info": {"split": split, "index": idx, "human_critique": human_critique, "ground_truth_choice": gt_choice},
        # template-required fields
        "response_A": resp_a,
        "response_B": resp_b,
        # optional: for process reward
        "human_critique": human_critique,
    }


def convert_split(raw_path: Path, split: str, max_items: int) -> List[Dict[str, Any]]:
    """Convert and subsample one split to list of dicts."""
    examples = list(load_jsonl_gz(raw_path))
    rng = random.Random(RNG_SEED)
    rng.shuffle(examples)
    examples = examples[:max_items]
    return [map_example(ex, split, i) for i, ex in enumerate(examples)]


def dump_json(data: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    # 路径硬编码（按你的环境固定）
    data_dir = Path("/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data/HelpSteer3_CLI/preference")
    out_dir = Path("/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data")

    # 仍保留可选的 size 参数，方便你临时改样本量
    parser.add_argument("--train_size", type=int, default=5000, help="Number of train samples")
    parser.add_argument("--test_size", type=int, default=500, help="Number of test samples")
    args = parser.parse_args()

    # 如果你希望未来恢复可配置路径，把下面两行替换回 argparse 的 data_dir/out_dir 即可
    # data_dir = Path(args.data_dir)
    # out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = data_dir / "train.jsonl.gz"
    val_raw = data_dir / "validation.jsonl.gz"
    if not train_raw.exists() or not val_raw.exists():
        raise FileNotFoundError("Expect train.jsonl.gz and validation.jsonl.gz inside data_dir")

    print("Converting train split…")
    train_data = convert_split(train_raw, "train", args.train_size)
    print(f"Train samples: {len(train_data)}")

    print("Converting test split…")
    test_data = convert_split(val_raw, "test", args.test_size)
    print(f"Test samples: {len(test_data)}")

    dump_json(train_data, out_dir / "train.json")
    dump_json(test_data, out_dir / "test.json")
    print("Saved JSONL to", out_dir)


if __name__ == "__main__":
    main()

