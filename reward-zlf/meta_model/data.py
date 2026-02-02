from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _normalize_choice(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"A", "0"}:
        return "A"
    if s in {"B", "1"}:
        return "B"
    if s in {"RESP1", "RESPONSE1"}:
        return "A"
    if s in {"RESP2", "RESPONSE2"}:
        return "B"
    return None


def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text(prompt_msgs: List[Dict[str, str]], response_text: str) -> str:
    # prompt is a list of chat messages already; keep it simple and robust.
    prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in (prompt_msgs or [])])
    return prompt + "\n" + response_text


@dataclass
class PairwiseBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


class PairwisePreferenceDataset(Dataset):
    def __init__(self, json_path: str):
        self.data = load_json_array(json_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]

        prompt = ex.get("prompt")
        resp_a = ex.get("response_A")
        resp_b = ex.get("response_B")

        gt = _normalize_choice(ex.get("ground_truth_choice"))
        if gt is None:
            rm = ex.get("reward_model") or {}
            gt = _normalize_choice(rm.get("ground_truth_choice") or rm.get("ground_truth"))

        if gt not in {"A", "B"}:
            # fallback: treat A as chosen
            gt = "A"

        if gt == "A":
            chosen = resp_a
            rejected = resp_b
        else:
            chosen = resp_b
            rejected = resp_a

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


class PairwiseCollator:
    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            # common choice for decoder-only models
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> PairwiseBatch:
        chosen_texts = [build_text(ex["prompt"], ex["chosen"]) for ex in batch]
        rejected_texts = [build_text(ex["prompt"], ex["rejected"]) for ex in batch]

        tok_c = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok_r = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return PairwiseBatch(
            chosen_input_ids=tok_c["input_ids"],
            chosen_attention_mask=tok_c["attention_mask"],
            rejected_input_ids=tok_r["input_ids"],
            rejected_attention_mask=tok_r["attention_mask"],
        )

