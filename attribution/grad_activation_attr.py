"""
Gradient x activation attribution for OLMo-2-1B on the country-capital ICL task.

For each matched (correct in both conditions) sample:
    metric = mean_t log P(gold_t | prefix, gold_{<t})       # teacher-forced
    metric.backward()
    For each transformer layer i:
        mlp_score[i, j]  = sum over seq of  A_mlp[1, s, j]  * A_mlp.grad[1, s, j]
        attn_score[i, h] = sum over (seq, head_dim) of
                           A_attn[1, s, h, d] * A_attn.grad[1, s, h, d]

Where:
    A_mlp  = input to down_proj  (the SwiGLU intermediate, [B, S, d_ff])
    A_attn = input to o_proj     (per-head outputs, [B, S, n_heads * d_head])

Final per-condition output averages scores across samples.

Outputs (in attribution/results/):
    attr_in_cxt.pt   {"mlp": [L, d_ff], "attn": [L, H], "metric": float, "n": int}
    attr_recall.pt   same shape

Run:
    python grad_activation_attr.py --condition both
    python grad_activation_attr.py --condition in_cxt --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ATTR_RESULTS_DIR, MODEL_NAME
from parse_results import Sample, get_matched_samples


# ---------------------------------------------------------------------------
# Model / tokenization
# ---------------------------------------------------------------------------

def load_model(model_name: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    # Param grads not needed; we only want activation grads. Saves memory.
    for p in model.parameters():
        p.requires_grad_(False)

    return tokenizer, model


def build_chat_prefix(tokenizer, icl_prompt: str) -> str:
    """Wrap the ICL prompt in the same chat template amao used at eval time."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": icl_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_with_gold(
    tokenizer, prefix_text: str, gold: str, device
) -> Tuple[torch.Tensor, List[int]]:
    """Tokenize prefix and (prefix + gold). Return (input_ids, gold_positions).

    gold_positions are indices in input_ids that correspond to gold tokens.
    Tries both no-separator and single-space-separator concatenation and
    keeps whichever yields prefix_ids as a clean prefix of full_ids.
    """
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids[0]
    L_pre = prefix_ids.shape[0]

    for sep in ("", " "):
        full_text = prefix_text + sep + gold
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
        if full_ids.shape[0] > L_pre and torch.equal(full_ids[:L_pre], prefix_ids):
            gold_positions = list(range(L_pre, full_ids.shape[0]))
            return full_ids.unsqueeze(0).to(device), gold_positions # here is the token positions

    raise ValueError(
        f"Could not find a clean prefix tokenization for gold={gold!r}. "
        f"Prefix tail tokens: {tokenizer.decode(prefix_ids[-5:])!r}"
    )


def tokenize_word_after_prefix(tokenizer, prefix_text: str, word: str) -> List[int]:
    """Return the token ids for `word` as it would tokenize right after prefix_text.

    Uses the same prefix-clean check as tokenize_with_gold so the resulting ids
    are the exact ones the model would emit at this position.
    """
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids[0]
    L_pre = prefix_ids.shape[0]
    for sep in ("", " "):
        full_ids = tokenizer(prefix_text + sep + word, return_tensors="pt").input_ids[0]
        if full_ids.shape[0] > L_pre and torch.equal(full_ids[:L_pre], prefix_ids):
            return full_ids[L_pre:].tolist()
    raise ValueError(
        f"Could not find a clean prefix tokenization for word={word!r}."
    )


# ---------------------------------------------------------------------------
# Activation cache via forward-pre-hooks
# ---------------------------------------------------------------------------

class ActivationCache:
    """Caches inputs to mlp.down_proj and self_attn.o_proj per layer.

    Calls .retain_grad() so each cached tensor's .grad is populated after
    metric.backward(). Cache is reset between samples by overwriting the
    dict entries (old tensors become unreferenced and free).
    """

    def __init__(self, model):
        self.model = model
        self.mlp_acts: dict[int, torch.Tensor] = {}
        self.attn_acts: dict[int, torch.Tensor] = {}
        self.handles = []

    def attach(self):
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            self.handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(self._mlp_hook(i))
            )
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._attn_hook(i))
            )
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self):
        self.mlp_acts.clear()
        self.attn_acts.clear()

    def _mlp_hook(self, i: int):
        def hook(_module, args):
            x = args[0]  # [B, S, d_ff] -- SwiGLU intermediate
            x.retain_grad()
            self.mlp_acts[i] = x
        return hook

    def _attn_hook(self, i: int):
        def hook(_module, args):
            x = args[0]  # [B, S, n_heads * d_head]
            x.retain_grad()
            self.attn_acts[i] = x
        return hook


# ---------------------------------------------------------------------------
# Metric and per-sample attribution
# ---------------------------------------------------------------------------

def last_token_gold_log_prob(
    logits: torch.Tensor, gold_positions: List[int], input_ids: torch.Tensor
) -> torch.Tensor:
    """Log P(last gold token | prefix, gold_{<t}). logits: [1, S, V]."""
    log_probs = logits.float().log_softmax(dim=-1)
    last_pos = gold_positions[-1]
    token_id = input_ids[0, last_pos]
    return log_probs[0, last_pos - 1, token_id]

def last_token_logit_diff(
    logits: torch.Tensor,
    gold_positions: List[int],
    input_ids: torch.Tensor,
    corrupted_token_ids: List[int],
) -> torch.Tensor:
    """logit(last gold token) - logit(last corrupted token) at the last gold position.

    For multi-token gold or corrupted words, uses the last token of each.
    logits: [1, S, V]
    """
    last_pos = gold_positions[-1]
    last_logits = logits[0, last_pos - 1].float()  # [V]
    gold_id = input_ids[0, last_pos]
    corrupted_id = corrupted_token_ids[-1]
    return last_logits[gold_id] - last_logits[corrupted_id]

def attribute_one_sample(
    model,
    cache: ActivationCache,
    input_ids: torch.Tensor,
    gold_positions: List[int],
    num_heads: int,
    head_dim: int,
    metric: str,
    corrupted_token_ids: List[int] | None = None,
):
    cache.clear()

    # Feed via inputs_embeds so the autograd graph runs through layers even
    # though all param grads are disabled.
    embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

    out = model(inputs_embeds=embeds, use_cache=False)
    if metric == "log_prob":
        m = last_token_gold_log_prob(out.logits, gold_positions, input_ids)
    elif metric == "logit_diff":
        m = last_token_logit_diff(
            out.logits, gold_positions, input_ids, corrupted_token_ids
        )
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
    m.backward()

    L = len(cache.mlp_acts)
    mlp_scores = []
    attn_scores = []
    for i in range(L):
        a = cache.mlp_acts[i]
        s_mlp = (a.detach().float() * a.grad.float()).sum(dim=(0, 1))  # [d_ff]
        mlp_scores.append(s_mlp.cpu())

        a = cache.attn_acts[i]
        B, S, _ = a.shape
        a_r = a.detach().float().view(B, S, num_heads, head_dim)
        g_r = a.grad.float().view(B, S, num_heads, head_dim)
        s_attn = (a_r * g_r).sum(dim=(0, 1, 3))  # [n_heads]
        attn_scores.append(s_attn.cpu())

    return (
        torch.stack(mlp_scores),  # [L, d_ff]
        torch.stack(attn_scores),  # [L, n_heads]
        m.detach().float().cpu().item(),
    )


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def run_condition(
    model,
    tokenizer,
    samples: List[Sample],
    num_heads: int,
    head_dim: int,
    limit: int | None,
    metric: str,
):
    cache = ActivationCache(model).attach()
    sum_mlp = None
    sum_attn = None
    sum_metric = 0.0
    n = 0

    try:
        loop = samples if limit is None else samples[:limit]
        device = next(model.parameters()).device
        for s in tqdm(loop):
            prefix = build_chat_prefix(tokenizer, s.prompt)
            input_ids, gold_positions = tokenize_with_gold(
                tokenizer, prefix, s.gold, device
            )
            corrupted_token_ids = (
                tokenize_word_after_prefix(tokenizer, prefix, s.corrupted)
                if metric == "logit_diff" else None
            )
            mlp_s, attn_s, m_val = attribute_one_sample(
                model, cache, input_ids, gold_positions, num_heads, head_dim,
                metric=metric, corrupted_token_ids=corrupted_token_ids,
            )
            mlp_s = mlp_s.abs()
            attn_s = attn_s.abs()
            sum_mlp = mlp_s if sum_mlp is None else sum_mlp + mlp_s
            sum_attn = attn_s if sum_attn is None else sum_attn + attn_s
            sum_metric += m_val
            n += 1
    finally:
        cache.detach()

    return {
        "mlp": sum_mlp / n,
        "attn": sum_attn / n,
        "metric": sum_metric / n,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=["in_cxt", "recall", "both"], default="both"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument(
        "--metric", choices=["log_prob", "logit_diff"], default="log_prob"
    )
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    tokenizer, model = load_model(MODEL_NAME, dtype)
    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
    print(
        f"Model: {MODEL_NAME}  layers={cfg.num_hidden_layers}  "
        f"d_model={cfg.hidden_size}  n_heads={num_heads}  d_head={head_dim}  "
        f"d_ff={cfg.intermediate_size}"
    )

    in_cxt_samples, recall_samples = get_matched_samples()
    print(f"Matched correct samples: {len(in_cxt_samples)}")

    targets = []
    if args.condition in ("in_cxt", "both"):
        targets.append(("in_cxt", in_cxt_samples))
    if args.condition in ("recall", "both"):
        targets.append(("recall", recall_samples))

    for name, samples in targets:
        print(f"\n=== Attribution: {name}  metric={args.metric} ===")
        result = run_condition(
            model, tokenizer, samples, num_heads, head_dim, args.limit, args.metric
        )
        out_path = ATTR_RESULTS_DIR / f"attr_{name}_{args.metric}.pt"
        torch.save(result, out_path)
        metric_label = (
            "mean log P(gold)" if args.metric == "log_prob"
            else "mean logit_diff(gold - corrupted)"
        )
        print(
            f"  n={result['n']}  {metric_label}={result['metric']:.4f}  "
            f"saved to {out_path}"
        )


if __name__ == "__main__":
    main()