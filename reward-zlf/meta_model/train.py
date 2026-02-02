from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PairwiseCollator, PairwisePreferenceDataset
from .model import MetaRewardModel


def pairwise_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    # -log sigmoid(r_chosen - r_rejected)
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def eval_win_rate(model: MetaRewardModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    wins = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            c_ids = batch.chosen_input_ids.to(device)
            c_mask = batch.chosen_attention_mask.to(device)
            r_ids = batch.rejected_input_ids.to(device)
            r_mask = batch.rejected_attention_mask.to(device)

            c = model(c_ids, c_mask).scores
            r = model(r_ids, r_mask).scores
            wins += (c > r).sum().item()
            total += c.numel()
    return float(wins) / float(max(1, total))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--val_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=4096)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model = MetaRewardModel.build(args.model_path, device=device, dtype=dtype)
    if args.freeze_backbone:
        model.freeze_backbone(True)

    train_ds = PairwisePreferenceDataset(args.train_json)
    val_ds = PairwisePreferenceDataset(args.val_json)

    collator = PairwiseCollator(args.model_path, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_win = -1.0
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            c_ids = batch.chosen_input_ids.to(device)
            c_mask = batch.chosen_attention_mask.to(device)
            r_ids = batch.rejected_input_ids.to(device)
            r_mask = batch.rejected_attention_mask.to(device)

            c = model(c_ids, c_mask).scores
            r = model(r_ids, r_mask).scores

            loss = pairwise_loss(c, r) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"[train] step={global_step}/{total_steps} loss={loss.item() * args.grad_accum:.6f} lr={lr:.2e}")

                if global_step % args.eval_every == 0:
                    win = eval_win_rate(model, val_loader, device)
                    print(f"[eval] step={global_step} win_rate={win:.4f}")
                    if win > best_win:
                        best_win = win
                        print(f"[ckpt] new best win_rate={best_win:.4f}, saving to {args.output_dir}")
                        model.save(args.output_dir)

        # end epoch eval
        win = eval_win_rate(model, val_loader, device)
        print(f"[eval] epoch_end epoch={epoch} win_rate={win:.4f}")
        if win > best_win:
            best_win = win
            print(f"[ckpt] new best win_rate={best_win:.4f}, saving to {args.output_dir}")
            model.save(args.output_dir)

    # save final
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save(final_dir)
    with open(os.path.join(final_dir, "train_args.txt"), "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()

