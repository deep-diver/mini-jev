"""
Sweep jev-style prompts on real alphaXiv titles.

The point of scoring with AUC rather than accuracy: a constant Yes-bias does
not move a ranking. If a variant is at 0.5 the prompt is not merely
mis-calibrated, the ordering carries nothing. So a variant that lifts AUC is a
real fix, and one that only lifts accuracy is just moving a threshold.

Every variant is a single forward pass reading two candidate tokens -- the jev
structure is held fixed and only the wording, the option set and the in-context
examples change.

    python3 scripts/prompt_sweep.py
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "data", "eval", "alphaxiv_labels.json")
MODEL = "google/gemma-3-270m-it"
CRIT = "This is about reinforcement learning, agents, or training with rewards."

# Shots are written here rather than taken from the eval set, which would leak.
SHOT_POS = ["PPO-Zero: Policy Optimization without a Value Network",
            "WebPilot: An Autonomous Agent for Long-Horizon Web Tasks",
            "Reward Shaping for Sparse-Signal Robotic Control"]
SHOT_NEG = ["Sheaf Cohomology of Toric Varieties over Finite Fields",
            "FastSplat: Real-Time Gaussian Splatting for Mobile GPUs",
            "A Survey of Tokenization Methods for Multilingual Corpora"]


def auc(scores, y):
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    return sum((a > b) + 0.5 * (a == b)
               for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only N titles")
    args = ap.parse_args()

    lab = json.load(open(LABELS))
    if args.limit:
        lab = lab[:args.limit]
    titles = [x["title"] for x in lab]
    y = np.array([x["label"] for x in lab], dtype=bool)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()

    def ids(*words):
        return [tok.encode(w, add_special_tokens=False)[0] for w in words]

    def chat(content, prefix=""):
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True) + prefix

    YES, NO = ids("Yes", "yes"), ids("No", "no")
    TRUE, FALSE = ids("true", "True"), ids("false", "False")
    A, B = ids("A"), ids("B")
    ONE, ZERO = ids("1"), ids("0")

    def label_block(n, pos_label, neg_label):
        # Plain pattern completion: no chat template, no instruction, just
        # Title/Label pairs for the model to continue.
        return "".join(f'Title: "{SHOT_POS[i]}"\nLabel: {pos_label}\n\n'
                       f'Title: "{SHOT_NEG[i]}"\nLabel: {neg_label}\n\n'
                       for i in range(n))

    def shots(n, pos_label, neg_label):
        out = []
        for i in range(n):
            out.append((SHOT_POS[i], pos_label))
            out.append((SHOT_NEG[i], neg_label))
        return out

    # name -> (prompt builder, positive ids, negative ids, invert)
    VARIANTS = {
        "0 현재 배포본 (State/Context)": (
            lambda t: chat(f'State / Context:\nElement: <a>\nText: "{t}"\n\n'
                           f'Question:\nThe user is looking for: {CRIT}\n\n'
                           f'Does this page element match what the user is looking for?\n\n'
                           f'Answer with Yes or No only.'), YES, NO, False),
        "1 최소형": (
            lambda t: chat(f'Title: {t}\nTopic: {CRIT}\nMatch? Answer Yes or No.'),
            YES, NO, False),
        "2 분류 지시": (
            lambda t: chat(f'Classify this paper title.\n\nTitle: "{t}"\n\n'
                           f'Is it about reinforcement learning, agents, or training '
                           f'with rewards?\nAnswer with Yes or No only.'), YES, NO, False),
        "3 직설 (기준문 제거)": (
            lambda t: chat(f'"{t}"\n\nDoes this paper use reinforcement learning or '
                           f'build an autonomous agent? Yes or No.'), YES, NO, False),
        "4 극성 반전 (무관한가?)": (
            lambda t: chat(f'Title: "{t}"\n\nThe user is looking for: {CRIT}\n\n'
                           f'Is this title UNRELATED to what the user wants? Yes or No.'),
            YES, NO, True),
        "5 2-shot": (
            lambda t: chat("Label each paper title Yes or No.\nQuestion: is it about "
                           f"{CRIT}\n\n" +
                           "".join(f'Title: "{s}"\nAnswer: {l}\n\n'
                                   for s, l in shots(1, "Yes", "No")) +
                           f'Title: "{t}"\nAnswer:'), YES, NO, False),
        "6 6-shot": (
            lambda t: chat("Label each paper title Yes or No.\nQuestion: is it about "
                           f"{CRIT}\n\n" +
                           "".join(f'Title: "{s}"\nAnswer: {l}\n\n'
                                   for s, l in shots(3, "Yes", "No")) +
                           f'Title: "{t}"\nAnswer:'), YES, NO, False),
        "7 6-shot + 접두사 강제": (
            lambda t: chat("Label each paper title Yes or No.\nQuestion: is it about "
                           f"{CRIT}\n\n" +
                           "".join(f'Title: "{s}"\nAnswer: {l}\n\n'
                                   for s, l in shots(3, "Yes", "No")) +
                           f'Title: "{t}"', prefix="\nAnswer:"), YES, NO, False),
        "8 완성형 (챗 템플릿 없음)": (
            lambda t: (f'Question: is this paper about {CRIT}\n\n'
                       f'Title: "{SHOT_POS[0]}"\nAnswer: Yes\n\n'
                       f'Title: "{SHOT_NEG[0]}"\nAnswer: No\n\n'
                       f'Title: "{t}"\nAnswer:'), YES, NO, False),
        "9 A/B 선택": (
            lambda t: chat(f'Title: "{t}"\n\nWhich describes it?\n'
                           f'A. It is about {CRIT}\nB. It is about something else.\n\n'
                           f'Answer with the letter only.'), A, B, False),
        "10 A/B 순서 뒤집음": (
            lambda t: chat(f'Title: "{t}"\n\nWhich describes it?\n'
                           f'A. It is about something else.\nB. It is about {CRIT}\n\n'
                           f'Answer with the letter only.'), A, B, True),
        "11 JSON 접두사": (
            lambda t: chat(f'Paper title: "{t}"\n\nThe user is looking for: {CRIT}\n\n'
                           f'Reply with JSON only.', prefix='{"relevant": '),
            TRUE, FALSE, False),
        "12 1/0 비트": (
            lambda t: chat(f'Title: "{t}"\nTopic: {CRIT}\n\n'
                           f'Output 1 if it matches, 0 if not. Output the digit only.'),
            ONE, ZERO, False),

        # The three changes below are what actually move the number, and each
        # one removes a piece of "instruction following" rather than improving
        # it. See the ablation printed at the end.
        "13 패턴완성 RL/other + 지시문": (
            lambda t: ("Label each paper title with RL or other.\n\n"
                       + label_block(2, "RL", "other") + f'Title: "{t}"\nLabel:'),
            ids("RL"), ids("other"), False),
        "14 패턴완성 RL/other, 지시문 없음": (
            lambda t: label_block(2, "RL", "other") + f'Title: "{t}"\nLabel:',
            ids("RL"), ids("other"), False),
        "15 (14) + 챗 템플릿 되돌림": (
            lambda t: chat("Label each paper title with RL or other.\n\n"
                           + label_block(2, "RL", "other") + f'Title: "{t}"',
                           prefix="\nLabel:"),
            ids("RL"), ids("other"), False),
    }

    @torch.no_grad()
    def score(build, pos, neg, invert):
        prompts = [build(t) for t in titles]
        out = []
        for i in range(0, len(prompts), 16):
            # A chat-templated prompt already carries <bos>; a raw completion
            # does not, so let the tokenizer add it in that case.
            raw = not prompts[0].startswith(tok.bos_token or "<bos>")
            enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True,
                      add_special_tokens=raw).to(device)
            last = enc["attention_mask"].sum(1) - 1
            rows = torch.arange(len(last), device=device)
            lg = model(**enc).logits[rows, last].float()
            a = lg[:, pos].max(1).values
            b = lg[:, neg].max(1).values
            out += torch.softmax(torch.stack([a, b], 1), 1)[:, 0].tolist()
        return [1.0 - s for s in out] if invert else out

    print(f"{len(titles)}편 · 양성 {int(y.sum())} ({y.mean():.0%}) · {MODEL} · {device}\n")
    print(f"{'프롬프트 변형':30s} {'AUC':>6s} {'평균점수':>8s} {'표준편차':>8s}")
    print("-" * 56)
    rows = []
    for name, (build, pos, neg, inv) in VARIANTS.items():
        s = score(build, pos, neg, inv)
        a = auc(s, y)
        rows.append((a, name, float(np.mean(s)), float(np.std(s))))
        print(f"{name:30s} {a:6.3f} {np.mean(s):8.3f} {np.std(s):8.4f}")

    best = max(rows)
    print("-" * 56)
    print(f"최고: {best[1].strip()}  AUC {best[0]:.3f}")
    print(f"(0.500 = 순위 정보 없음.  은닉상태 직접 프로브는 같은 데이터에서 0.839)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
