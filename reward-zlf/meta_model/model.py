from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class MetaRewardModelOutput:
    scores: torch.Tensor  # (batch,)


class MetaRewardModel(nn.Module):
    """Pairwise reward model: base transformer + scalar reward head.

    Notes:
        - We use the hidden state of the last *non-masked* token (based on attention_mask)
          as the pooled representation.
        - This avoids relying on pad_token_id (many base LMs don't define it).
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from config")

        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    @classmethod
    def build(cls, model_path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        model = cls(model_path)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> MetaRewardModelOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state  # (b, s, h)

        # last token index per sample
        # attention_mask is 1 for real tokens
        seq_lens = attention_mask.long().sum(dim=-1).clamp(min=1)
        last_idx = (seq_lens - 1).to(last_hidden_state.device)

        bsz = input_ids.shape[0]
        pooled = last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), last_idx]  # (b, h)

        scores = self.reward_head(pooled).squeeze(-1)  # (b,)
        return MetaRewardModelOutput(scores=scores)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def save(self, output_dir: str) -> None:
        # save backbone in HF format
        self.backbone.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        # save reward head separately
        torch.save({"reward_head": self.reward_head.state_dict()}, f"{output_dir}/reward_head.pt")

    @classmethod
    def load(cls, model_dir: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        model = cls(model_dir)
        ckpt = torch.load(f"{model_dir}/reward_head.pt", map_location="cpu")
        model.reward_head.load_state_dict(ckpt["reward_head"], strict=True)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        return model
