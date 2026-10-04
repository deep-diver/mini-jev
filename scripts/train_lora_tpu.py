"""
LoRA + pointer head on Gemma 3 270M, in JAX, on a TPU.

The frozen-backbone head reached 0.704 out of domain against 0.485 for reading
Yes/No logits off the same model. The rest of what kev does is to let the
backbone move: a low-rank adapter on every attention projection, trained
together with the head on the option distribution.

Self-contained on purpose -- it loads the weights through transformers and
converts them once, so a TPU VM needs this file, the data and the model cache,
and nothing else from the repo.

    python3 scripts/train_lora_tpu.py --steps 3000 --batch 32
"""

import argparse
import functools
import json
import os
import sys
import time

import numpy as np

# Gemma 3 270M, matching the repo's JAX port.
HIDDEN_SIZE = 640
NUM_LAYERS = 18
NUM_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 256
RMS_NORM_EPS = 1e-6
EMBED_SCALE = HIDDEN_SIZE ** 0.5
ATTN_SCALING = HEAD_DIM ** -0.5
SLIDING_WINDOW = 512
LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"] + \
              ["sliding_attention"] * 5 + ["full_attention"] + \
              ["sliding_attention"] * 5 + ["full_attention"]
LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj")


def render(state, instructions, options):
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


# --------------------------------------------------------------------------

def build_batchable(tok, records, max_len, max_options):
    """Tokenise once; keep the marker positions the pointer head reads."""
    ids = np.zeros((len(records), max_len), dtype=np.int32)
    lens = np.zeros(len(records), dtype=np.int32)
    marks = np.zeros((len(records), max_options), dtype=np.int32)
    mask = np.zeros((len(records), max_options), dtype=bool)
    gold = np.zeros(len(records), dtype=np.int32)
    kept = []

    for r in records:
        text = render(r["state"], r["instructions"], r["options"])
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True,
                  truncation=True, max_length=max_len)
        n = len(enc["input_ids"])
        # A truncated sequence loses its markers, so drop it rather than train
        # the head to point at padding.
        pos = []
        ok = True
        for i in range(len(r["options"])):
            ch = text.find(f"<{i}>")
            p = token_at(enc["offset_mapping"], ch + 2, n)
            if ch < 0 or p is None or p >= n:
                ok = False
                break
            pos.append(p)
        if not ok:
            continue
        j = len(kept)
        ids[j, :n] = enc["input_ids"]
        lens[j] = n
        marks[j, :len(pos)] = pos
        mask[j, :len(pos)] = True
        gold[j] = r["gold"]
        kept.append(r)

    k = len(kept)
    return {"ids": ids[:k], "lens": lens[:k], "marks": marks[:k],
            "mask": mask[:k], "gold": gold[:k]}, kept


def token_at(offsets, char, n):
    for i, (a, b) in enumerate(offsets[:n]):
        if a <= char < b:
            return i
    return None


# --------------------------------------------------------------------------

def load_weights(model_id, jnp):
    """Prefer the safetensors file: a TPU VM then needs no torch at all."""
    import os
    if os.path.isdir(model_id):
        p = os.path.join(model_id, "model.safetensors")
    else:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(model_id, "model.safetensors")
    if isinstance(p, str) and os.path.exists(p):
        from safetensors import safe_open
        out = {}
        with safe_open(p, framework="np") as f:
            for k in f.keys():
                out[k] = jnp.asarray(f.get_tensor(k), dtype=jnp.float32)
        return out
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    return {k: jnp.asarray(v.detach().numpy()) for k, v in m.state_dict().items()}


def stack_layers(weights, jnp):
    """Stack the 18 per-layer tensors into one array each.

    Unrolling the layers in Python made an HLO large enough that XLA was killed
    by the host OOM killer while compiling the backward pass -- at batch 4 and
    128 tokens, so it was the graph, not the activations. Scanning compiles one
    layer body instead of eighteen.
    """
    names = ["input_layernorm.weight", "post_attention_layernorm.weight",
             "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
             "self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "self_attn.q_norm.weight", "self_attn.k_norm.weight",
             "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    return {n: jnp.stack([weights[f"model.layers.{i}.{n}"]
                          for i in range(NUM_LAYERS)]) for n in names}


def make_model(jnp, jax):
    def rms_norm(x, w, eps=RMS_NORM_EPS):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(var + eps)) * (1.0 + w)

    def rotate_half(x):
        h = x.shape[-1] // 2
        return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)

    def rope(seq_len, head_dim, base):
        inv = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        f = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv)
        emb = jnp.concatenate([f, f], axis=-1)
        return jnp.cos(emb)[None, None], jnp.sin(emb)[None, None]

    def dense(x, W, a, b, scale):
        return jnp.dot(x, W.T) + scale * jnp.dot(jnp.dot(x, a.T), b.T)

    def backbone(stacked, lora, embed, final_norm, input_ids, real_lens,
                 max_len, scale):
        batch = input_ids.shape[0]
        x = embed[input_ids] * EMBED_SCALE
        cos_f, sin_f = rope(max_len, HEAD_DIM, 1000000.0)
        cos_s, sin_s = rope(max_len, HEAD_DIM, 10000.0)
        q_idx = jnp.arange(max_len)[None, :, None]
        k_idx = jnp.arange(max_len)[None, None, :]
        causal = (q_idx >= k_idx) & (k_idx < real_lens[:, None, None])
        sliding = causal & ((q_idx - k_idx) <= SLIDING_WINDOW)
        is_full = jnp.array([t == "full_attention" for t in LAYER_TYPES])

        @jax.checkpoint
        def layer(x, p):
            w, lo, full = p
            cos = jnp.where(full, cos_f, cos_s)
            sin = jnp.where(full, sin_f, sin_s)
            m = jnp.where(full, causal, sliding)

            res = x
            h = rms_norm(x, w["input_layernorm.weight"])
            q = dense(h, w["self_attn.q_proj.weight"],
                      lo["q_A"], lo["q_B"], scale)
            k = dense(h, w["self_attn.k_proj.weight"],
                      lo["k_A"], lo["k_B"], scale)
            v = dense(h, w["self_attn.v_proj.weight"],
                      lo["v_A"], lo["v_B"], scale)
            q = q.reshape(batch, max_len, NUM_HEADS, HEAD_DIM).swapaxes(1, 2)
            k = k.reshape(batch, max_len, NUM_KV_HEADS, HEAD_DIM).swapaxes(1, 2)
            v = v.reshape(batch, max_len, NUM_KV_HEADS, HEAD_DIM).swapaxes(1, 2)
            q = rms_norm(q, w["self_attn.q_norm.weight"])
            k = rms_norm(k, w["self_attn.k_norm.weight"])
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            sc = jnp.matmul(q, k.swapaxes(-1, -2)) * ATTN_SCALING
            sc = jnp.where(m[:, None, :, :], sc, -1e9)
            o = jnp.matmul(jax.nn.softmax(sc, axis=-1), v)
            o = o.swapaxes(1, 2).reshape(batch, max_len, NUM_HEADS * HEAD_DIM)
            o = dense(o, w["self_attn.o_proj.weight"], lo["o_A"], lo["o_B"], scale)
            x = res + rms_norm(o, w["post_attention_layernorm.weight"])

            res = x
            h = rms_norm(x, w["pre_feedforward_layernorm.weight"])
            gate = jnp.dot(h, w["mlp.gate_proj.weight"].T)
            up = jnp.dot(h, w["mlp.up_proj.weight"].T)
            d = jnp.dot(jax.nn.gelu(gate, approximate=True) * up,
                        w["mlp.down_proj.weight"].T)
            x = res + rms_norm(d, w["post_feedforward_layernorm.weight"])
            return x, None

        x, _ = jax.lax.scan(layer, x, (stacked, lora, is_full))
        # The final norm matters: without it the residual stream the head reads
        # has a norm around 24,000. With it this reproduces HuggingFace's last
        # hidden state (cosine 1.000000, max diff 2e-4).
        return rms_norm(x, final_norm)

    def logits_fn(params, const, batch, max_len, scale):
        x = backbone(const["stacked"], params["lora"], const["embed"],
                     const["final_norm"], batch["ids"], batch["lens"],
                     max_len, scale)
        rows = jnp.arange(x.shape[0])
        query = x[rows, batch["lens"] - 1]
        keys = x[rows[:, None], batch["marks"]]

        head = params["head"]
        gq = (1.0 + head["ln_w"]) * normalise(jnp, query) + head["ln_b"]
        gk = (1.0 + head["ln_w"]) * normalise(jnp, keys) + head["ln_b"]
        q = jnp.dot(gq, head["Wq"].T)
        k = jnp.dot(gk, head["Wk"].T)
        logit = jnp.einsum("bod,bd->bo", k, q) / (q.shape[-1] ** 0.5)
        return jnp.where(batch["mask"], logit, -1e4)

    # Both are returned so the backbone can be checked against HuggingFace on
    # its own; a silent mismatch here would invalidate every later number.
    return logits_fn, backbone


def normalise(jnp, x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True)
    return (x - mu) / (sd + eps)


# --------------------------------------------------------------------------

def init_params(weights, rank, dim, seed, jnp, jax):
    key = jax.random.PRNGKey(seed)
    lora = {}
    for short, full in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                        ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj")):
        out_f, in_f = weights[f"model.layers.0.{full}.weight"].shape
        key, k1 = jax.random.split(key)
        lora[f"{short}_A"] = (jax.random.normal(k1, (NUM_LAYERS, rank, in_f))
                              * (1.0 / in_f ** 0.5))
        lora[f"{short}_B"] = jnp.zeros((NUM_LAYERS, out_f, rank))  # starts a no-op
    key, k1, k2 = jax.random.split(key, 3)
    head = {
        "Wq": jax.random.normal(k1, (dim, HIDDEN_SIZE)) * (1.0 / HIDDEN_SIZE ** 0.5),
        "Wk": jax.random.normal(k2, (dim, HIDDEN_SIZE)) * (1.0 / HIDDEN_SIZE ** 0.5),
        "ln_w": jnp.zeros((HIDDEN_SIZE,)),
        "ln_b": jnp.zeros((HIDDEN_SIZE,)),
    }
    return {"lora": lora, "head": head}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/decision")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out", default="data/decision/lora_head.npz")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--max-options", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--no-lora", action="store_true",
                    help="train the head only, with the backbone untouched; "
                         "scale 0 removes the adapter from the forward pass and "
                         "zeroes its gradient, so the run is otherwise identical")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from transformers import AutoTokenizer

    print("devices:", jax.devices(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    weights = load_weights(args.model, jnp)
    if "model.embed_tokens.weight" not in weights:
        raise SystemExit("unexpected checkpoint layout: " + str(list(weights)[:5]))
    const = {"stacked": stack_layers(weights, jnp),
             "embed": weights["model.embed_tokens.weight"],
             "final_norm": weights["model.norm.weight"]}
    print(f"{len(weights)} frozen tensors, stacked into "
          f"{len(const['stacked'])} arrays", flush=True)

    def read(split):
        with open(os.path.join(args.data, f"{split}.jsonl")) as f:
            return [json.loads(l) for l in f]

    t0 = time.time()
    train, _ = build_batchable(tok, read("train"), args.max_len, args.max_options)
    dev, _ = build_batchable(tok, read("dev"), args.max_len, args.max_options)
    print(f"train {len(train['gold']):,} · dev {len(dev['gold']):,} "
          f"· tokenised in {time.time() - t0:.0f}s", flush=True)

    logits_fn, _ = make_model(jnp, jax)
    params = init_params(weights, args.rank, args.dim, 0, jnp, jax)
    n_train = sum(x.size for x in params["lora"].values())
    n_head = sum(x.size for x in params["head"].values())
    if args.no_lora:
        print(f"trainable: head only {n_head:,} (adapter disabled)", flush=True)
    else:
        print(f"trainable: LoRA {n_train:,} + head {n_head:,} = "
              f"{n_train + n_head:,} ({(n_train + n_head) / 268e6:.2%} of the model)",
              flush=True)

    scale = 0.0 if args.no_lora else args.alpha / args.rank
    warmup = max(1, min(100, args.steps // 10))
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, args.steps, 1e-5)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    def loss_fn(p, batch):
        lg = logits_fn(p, const, batch, args.max_len, scale)
        return optax.softmax_cross_entropy_with_integer_labels(lg, batch["gold"]).mean()

    @jax.jit
    def step(p, s, batch):
        loss, grads = jax.value_and_grad(loss_fn)(p, batch)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    @jax.jit
    def acc_fn(p, batch):
        lg = logits_fn(p, const, batch, args.max_len, scale)
        return (lg.argmax(-1) == batch["gold"]).mean()

    def take(d, idx):
        return {k: jnp.asarray(v[idx]) for k, v in d.items()}

    def dev_acc(p, n=512):
        out = []
        for i in range(0, min(n, len(dev["gold"])), args.batch):
            j = np.arange(i, min(i + args.batch, len(dev["gold"])))
            if len(j) < args.batch:
                break
            out.append(float(acc_fn(p, take(dev, j))))
        return float(np.mean(out))

    rng = np.random.default_rng(0)
    n = len(train["gold"])
    print(f"\n{'step':>6s} {'loss':>7s} {'dev':>6s} {'s/step':>7s}", flush=True)
    t0, best = time.time(), -1.0
    for s in range(1, args.steps + 1):
        idx = rng.integers(0, n, args.batch)
        params, state, loss = step(params, state, take(train, idx))
        if s % args.eval_every == 0 or s == 1:
            a = dev_acc(params)
            print(f"{s:6d} {float(loss):7.4f} {a:6.3f} "
                  f"{(time.time() - t0) / s:7.3f}", flush=True)
            if a > best:
                best = a
                flat = {f"lora.{k}": np.asarray(v)
                        for k, v in params["lora"].items()}
                flat.update({f"head.{k}": np.asarray(v)
                             for k, v in params["head"].items()})
                np.savez_compressed(args.out, rank=args.rank, alpha=args.alpha,
                                    dim=args.dim, dev_acc=a, **flat)
    print(f"\nbest dev {best:.3f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
