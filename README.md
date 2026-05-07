# icl-attribution
# Attribution

Gradient × activation attribution for OLMo-2-1B-Instruct on a 5-shot
country-capital ICL task, comparing two conditions: **in_cxt** (model
follows the in-context shots) vs **recall** (model uses the parametric
answer).

## How the attribution score is computed

For each sample (correct in both conditions):

1. Forward the ICL prompt + gold answer through the model.
2. Compute a scalar metric at the last gold position:
   - `log_prob`: `log P(gold_last | prefix, gold_<t)`
   - `logit_diff`: `logit(gold_last) − logit(corrupted_last)`
3. `metric.backward()` to get gradients on cached activations
   (inputs to `mlp.down_proj` and `self_attn.o_proj`, kept via
   `retain_grad()` in forward-pre-hooks).
4. Score each component as `activation × grad`, summed over sequence
   positions (and head_dim for attention):
   - **MLP neuron** `[L, d_ff]`: `sum_s a[s, j] * a.grad[s, j]`
   - **Attention head** `[L, H]`: reshape `o_proj` input to
     `[B, S, H, d_head]`, then `sum over (s, d) of a * a.grad`
5. Take `|score|`, accumulate across samples, divide by `n`.

Output per condition × metric: `results/attr_{condition}_{metric}.pt`
with `{"mlp": [L, d_ff], "attn": [L, H], "metric": float, "n": int}`.

## How to run

**1. Get matched samples** (correct in both conditions):

```bash
python parse_results.py
```

**2. Compute attribution scores:**

```bash
# default: both conditions, log_prob metric
python grad_activation_attr.py

# other metric
python grad_activation_attr.py --metric logit_diff

# smoke test
python grad_activation_attr.py --condition in_cxt --limit 10
```

Flags: `--condition {in_cxt, recall, both}`, `--metric {log_prob, logit_diff}`,
`--dtype {bf16, fp16, fp32}`, `--limit N`.

**3. Plot results** (writes to `results/plots/`):

```bash
python plot_attribution.py
python plot_attribution.py --metric logit_diff
```

Flags: `--metric {log_prob, logit_diff}`, `--topk` (top-K MLP neurons per
layer in heatmap), `--topn` (global top-N for scatter and CSV).
