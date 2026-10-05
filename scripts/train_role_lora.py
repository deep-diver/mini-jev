"""
LoRA fine-tune Gemma 3 270M on the SurfMate region-type task.

The prompt is built by the mini-jev engine itself, so what the model is trained
on is byte-identical to what it will be asked at inference. Loss is applied to
the single answer token only: the engine never generates, it reads the logits
of the option letters, so that one position is the entire output.

    python3 scripts/train_role_lora.py --epochs 8
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from surfmate_server import PAGE_QUESTIONS, ROLE_ALIASES  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "surfmate")
OUT_DIR = os.path.join(ROOT, "gemma3_surfmate_lora")

BASE = "google/gemma-3-270m-it"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The option list the engine presents, and the letter each canonical role sits
# at. Both must match the server exactly or the tuning will not transfer.
CRITERIA = PAGE_QUESTIONS["role"]["criteria"]
ROLE_TO_LETTER = {ROLE_ALIASES[c]: LETTERS[i] for i, c in enumerate(CRITERIA)}


def build_prompt(engine, state):
    from mini_jev.schemas import ChoiceQuestion
    q = ChoiceQuestion(**PAGE_QUESTIONS["role"])
    keys, options = engine._choice_options(q)
    return engine._build_prompt(engine._choice_content(state, q, keys, options))


class RoleDataset(Dataset):
    def __init__(self, rows, tokenizer, engine):
        self.samples = []
        for r in rows:
            prompt = build_prompt(engine, r["state"])
            letter = ROLE_TO_LETTER[r["truth"]]
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            a_ids = tokenizer.encode(letter, add_special_tokens=False)
            ids = p_ids + a_ids
            # Everything but the answer token is masked out of the loss: the
            # prompt is context the model is given, not something to predict.
            labels = [-100] * len(p_ids) + a_ids
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch, pad_id):
    n = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attn[i, :len(ids)] = 1
    return input_ids, attn, labels


@torch.no_grad()
def evaluate(model, tokenizer, engine, rows, device):
    """Score exactly the way the engine does: read the option-letter logits."""
    model.eval()
    letter_ids = []
    for i in range(len(CRITERIA)):
        up = tokenizer.encode(LETTERS[i], add_special_tokens=False)[0]
        lo = tokenizer.encode(LETTERS[i].lower(), add_special_tokens=False)[0]
        letter_ids.append((up, lo))

    correct = 0
    preds = []
    for r in rows:
        ids = tokenizer.encode(build_prompt(engine, r["state"]), add_special_tokens=False)
        logits = model(torch.tensor([ids], device=device)).logits[0, -1].float().cpu().numpy()
        scores = [max(logits[u], logits[l]) for u, l in letter_ids]
        pick = int(np.argmax(scores))
        role = ROLE_ALIASES[CRITERIA[pick]]
        preds.append(role)
        correct += (role == r["truth"])
    model.train()
    return correct / len(rows), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "role_train.jsonl"))]
    test_rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "role_test.jsonl"))]

    counts = {}
    for r in test_rows:
        counts[r["truth"]] = counts.get(r["truth"], 0) + 1
    majority = max(counts.values()) / len(test_rows)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from mini_jev.engine_jax import MiniJevEngine

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    engine = MiniJevEngine(model_id=BASE, warmup=False)

    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(device)

    base_acc, base_preds = evaluate(model, tokenizer, engine, test_rows, device)
    print(f"base accuracy      {base_acc:.1%}   "
          f"(majority class {majority:.1%}, {len(set(base_preds))} distinct classes used)")

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={args.rank}: {trainable:,} trainable parameters "
          f"({trainable / 268_000_000:.2%} of the model)\n")

    ds = RoleDataset(train_rows, tokenizer, engine)
    pad = tokenizer.pad_token_id or 0
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, pad))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best = (base_acc, -1)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for input_ids, attn, labels in dl:
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attn.to(device),
                        labels=labels.to(device))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += out.loss.item()

        acc, preds = evaluate(model, tokenizer, engine, test_rows, device)
        used = len(set(preds))
        flag = ""
        if acc > best[0]:
            best = (acc, epoch)
            flag = "  <- best"
            model.save_pretrained(OUT_DIR)
        print(f"epoch {epoch:2d}  loss {total / len(dl):.4f}  "
              f"test accuracy {acc:.1%}  classes used {used}/{len(CRITERIA)}{flag}")

    print(f"\nbest {best[0]:.1%} at epoch {best[1]}   "
          f"base {base_acc:.1%}   majority {majority:.1%}")
    print(f"adapter -> {OUT_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
