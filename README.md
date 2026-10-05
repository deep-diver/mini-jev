# mini-jev

> A System One decision interface on Google Gemma 3 270M.
> Inspired by [TypeSafe AI's Jev](https://typesafe.ai).

Jev does not write text. You give it a state, declare the fields you want, and it
returns those fields with a probability attached. No generation means no parse
error and no invented field.

This repository builds that interface on a 270M model, and measures honestly
what it takes to make it work at that size.

| Approach | AUC on an unseen task |
|---|---|
| Reading Yes/No logits off the stock model | 0.485 |
| Frozen backbone + a trained pointer head | 0.704 |
| **LoRA + a trained pointer head** | **0.821** |

0.485 is a coin flip. The gap between the first and last row is the whole point.

## Start here

Two editions of the same notebook; the English one prints in English.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deep-diver/mini-jev/blob/main/notebooks/gemma3_270m_jev_en.ipynb)
&nbsp;English &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deep-diver/mini-jev/blob/main/notebooks/gemma3_270m_jev.ipynb)
&nbsp;한국어

Sections 1–4 run on a free CPU runtime in about fifteen minutes. Sections 5–7
need a GPU or a TPU.

## The three primitives

The interface is the same as Jev's, and each call is a single forward pass that
reads the logits of a fixed set of candidate tokens.

- **`noul`** — a yes probability, `p ∈ [0, 1]`
- **`choice`** — one option from a declared list, with the distribution over all of them
- **`score`** — a level on an ordered scale, as an expected value over digit tokens

## What the measurements say

**Reading Yes/No logits off a stock Gemma 3 270M does not work.** Over 1,200
forward passes it scores AUC 0.575, and on real page text 0.485. The cause is not
the readout: `Yes` and `No` hold 100% of the probability mass at the answer
position and the model picks the wrong one. Generation is fine — it writes
`Paris.` and `4` correctly — and asking for JSON instead returns
`{"relevant": true}` for all 179 test items. The output channel is saturated and
no output format gets around it.

**The information is there.** A 641-parameter linear probe on the mid-layer
residual stream scores around 0.97 leave-one-criterion-out on the same data. The
model knows; it cannot say.

**So train the readout.** A pointer head scores each option's marker against the
decision token, which lets the candidate set change per call, and a LoRA adapter
on the attention projections lets the backbone move with it. 1,803,520 trainable
parameters — 0.67% of the model.

Held out by construction — the training mix is news, encyclopedia entries, forum
questions, bank messages and reviews, with no paper text anywhere — the same
alphaXiv task goes **0.485 → 0.821**. Controlled on everything but the adapter:

| | In-domain dev | Unseen task |
|---|---|---|
| head only | 0.695 | 0.475 |
| LoRA + head | 0.877 | 0.821 |

**Longer is not better.** At 6000 steps in-domain dev rises to 0.887 while the
unseen task falls to 0.724. Selecting a checkpoint on in-domain dev picks the one
that transfers worst.

**The negatives decide the result.** A negative is drawn from the same corpus as
its positive, from a different class. Fill the negatives with boilerplate instead
and the model learns "has content" rather than "is on topic" — which scores well
in training and is flat on a page where every item is a paper title.

## Scripts

| | |
|---|---|
| [`build_decision_data.py`](scripts/build_decision_data.py) | 43,629 typed requests from nine public corpora |
| [`pointer_head.py`](scripts/pointer_head.py) | Frozen backbone, trained head, fitted temperature |
| [`train_lora_tpu.py`](scripts/train_lora_tpu.py) | LoRA + head in JAX, layers folded with `lax.scan` |
| [`eval_lora.py`](scripts/eval_lora.py) | Scores a checkpoint on the held-out task |
| [`tpu_lora.sh`](scripts/tpu_lora.sh) | Provisions a TPU and runs the whole cycle |
| [`prompt_sweep.py`](scripts/prompt_sweep.py) | 16 prompt variants, to show prompting is not the fix |
| [`auto_labels.py`](scripts/auto_labels.py) | Deriving label words from the criterion |
| [`eval_alphaxiv.py`](scripts/eval_alphaxiv.py) | Precision and recall against the hand labels |

## Training

```bash
python3 scripts/build_decision_data.py

python3 scripts/train_lora_tpu.py --model google/gemma-3-270m-it \
    --data data/decision --out lora.npz \
    --steps 2000 --batch 32 --max-len 256 --rank 16 --dim 256

python3 scripts/eval_lora.py --ckpt lora.npz --data data/eval/alphaxiv_multi.json
```

On a TPU, `scripts/tpu_lora.sh` handles one step per command. It pushes the
weights from the local Hugging Face cache, so the VM needs no token for the gated
repository.

```bash
scripts/tpu_lora.sh create && scripts/tpu_lora.sh setup && scripts/tpu_lora.sh push
scripts/tpu_lora.sh train && scripts/tpu_lora.sh log
scripts/tpu_lora.sh eval && scripts/tpu_lora.sh fetch
scripts/tpu_lora.sh stop
```

| Hardware | s/step | 2000 steps |
|---|---|---|
| TPU v5e-1 (batch 32) | 0.074 | 3 min |
| T4 (batch 16) | 0.95 | 32 min |

Being JAX, the same code runs on both. About ten minutes end to end on a TPU,
including provisioning.

## Evaluation data

[`data/eval/`](data/eval) holds 179 alphaXiv paper titles labelled by hand
against three criteria — reinforcement learning and agents, video and images,
mathematics and physics. Nothing resembling it appears in training, which is what
makes it worth measuring against. Every score is in
`alphaxiv_labels.json` alongside its label, so a run can be audited.

## Layout

```
notebooks/
  _source.txt, _source_en.txt    prose and code, reviewable in a diff
  build.py                       emits the .ipynb files
  gemma3_270m_jev*.ipynb         generated; do not edit directly
scripts/                         data, training, evaluation, TPU
data/eval/                       179 titles x 3 criteria, hand labelled
data/decision/lora_best.npz      the trained adapter and head, 6.4 MB
```

Editing a notebook means editing its `_source` file and running
`python3 notebooks/build.py`.

## Requirements

`transformers`, `torch` and `numpy` for the notebook; `jax`, `optax` and
`safetensors` for training. Gemma 3 270M is gated — accept the licence on the
[model page](https://huggingface.co/google/gemma-3-270m-it) first.
