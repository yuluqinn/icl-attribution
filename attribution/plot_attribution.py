"""
Visualize attribution scores from grad_activation_attr.py.

Inputs:
    results/attr_in_cxt.pt
    results/attr_recall.pt

Outputs (in results/plots/):
    attn_heatmaps.png       3-panel [L, H] heatmaps (in_cxt, recall, diff).
    mlp_topk_heatmaps.png   3-panel [L, K] heatmaps of MLP neurons selected
                            per layer by |in_cxt - recall|. Same neuron set
                            across panels so columns are comparable.
    mlp_layer_summary.png   Per-layer sum |score| bar charts (MLP and attn).
    top_neurons.csv         Global top-N (layer, neuron) by (in_cxt - recall).

Run:
    python plot_attribution.py
    python plot_attribution.py --topk 64 --topn 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from config import ATTR_RESULTS_DIR


PLOTS_DIR = ATTR_RESULTS_DIR / "plots"


def load_attr(metric: str):
    in_cxt = torch.load(ATTR_RESULTS_DIR / f"attr_in_cxt_{metric}.pt", weights_only=False)
    recall = torch.load(ATTR_RESULTS_DIR / f"attr_recall_{metric}.pt", weights_only=False)
    return in_cxt, recall


def diverging_norm(*tensors):
    """Symmetric diverging norm centered at 0 across all given tensors."""
    m = max(float(torch.abs(t).max().clamp_min(1e-12)) for t in tensors)
    return TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)


def plot_attn(in_cxt_attn, recall_attn, save_path: Path):
    diff = in_cxt_attn - recall_attn
    norm_pair = diverging_norm(in_cxt_attn, recall_attn)
    norm_diff = diverging_norm(diff)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (axes[0], in_cxt_attn, "in_cxt", norm_pair),
        (axes[1], recall_attn, "recall", norm_pair),
        (axes[2], diff, "in_cxt - recall", norm_diff),
    ]
    for ax, data, title, norm in panels:
        im = ax.imshow(data.numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(title)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Attention head attribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_topk(in_cxt_mlp, recall_mlp, k: int, save_path: Path):
    """Per layer, select top-K neurons by |diff|; show across 3 panels."""
    diff = in_cxt_mlp - recall_mlp
    topk_idx = torch.abs(diff).topk(k, dim=1).indices  # [L, K]

    def gather_topk(t):
        return torch.gather(t, 1, topk_idx)

    in_cxt_top = gather_topk(in_cxt_mlp)
    recall_top = gather_topk(recall_mlp)
    diff_top = gather_topk(diff)

    norm_pair = diverging_norm(in_cxt_top, recall_top)
    norm_diff = diverging_norm(diff_top)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (axes[0], in_cxt_top, "in_cxt", norm_pair),
        (axes[1], recall_top, "recall", norm_pair),
        (axes[2], diff_top, "in_cxt - recall", norm_diff),
    ]
    for ax, data, title, norm in panels:
        im = ax.imshow(data.numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(title)
        ax.set_xlabel(f"top-{k} neuron rank within layer (by |diff|)")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle(f"MLP neuron attribution: top-{k} per layer by |in_cxt - recall|")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_summary(in_cxt_mlp, recall_mlp, in_cxt_attn, recall_attn, save_path: Path):
    L = in_cxt_mlp.shape[0]
    layers = np.arange(L)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    width = 0.4

    axes[0].bar(layers - width / 2, torch.abs(in_cxt_mlp).sum(1).numpy(), width, label="in_cxt")
    axes[0].bar(layers + width / 2, torch.abs(recall_mlp).sum(1).numpy(), width, label="recall")
    axes[0].set_title("MLP: sum |neuron score| per layer")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("sum |attribution|")
    axes[0].legend()

    axes[1].bar(layers - width / 2, torch.abs(in_cxt_attn).sum(1).numpy(), width, label="in_cxt")
    axes[1].bar(layers + width / 2, torch.abs(recall_attn).sum(1).numpy(), width, label="recall")
    axes[1].set_title("Attn: sum |head score| per layer")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("sum |attribution|")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_top_neurons(in_cxt_mlp, recall_mlp, n: int, save_path: Path):
    diff = in_cxt_mlp - recall_mlp
    d_ff = diff.shape[1]
    flat = diff.flatten()
    topn_idx = torch.abs(flat).topk(n).indices

    rows = []
    for idx in topn_idx.tolist():
        layer = idx // d_ff
        neuron = idx % d_ff
        rows.append({
            "layer": layer,
            "neuron": neuron,
            "in_cxt": in_cxt_mlp[layer, neuron].item(),
            "recall": recall_mlp[layer, neuron].item(),
            "diff": diff[layer, neuron].item(),
        })

    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "neuron", "in_cxt", "recall", "diff"])
        w.writeheader()
        w.writerows(rows)


def iou_at_k(scores_a, scores_b, k: int) -> float:
    """IoU of top-k indices by |score| from two same-shape tensors."""
    flat_a = torch.abs(scores_a).flatten()
    flat_b = torch.abs(scores_b).flatten()
    top_a = set(torch.topk(flat_a, k).indices.tolist())
    top_b = set(torch.topk(flat_b, k).indices.tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def plot_iou_curve(in_cxt_mlp, recall_mlp, in_cxt_attn, recall_attn, save_path: Path):
    """Raw IoU(top-K) of in_cxt vs recall, swept over K, separately for heads
    and MLP neurons. Random-baseline IoU ≈ K / (2N − K) drawn as reference.
    Y-axis pinned to [0, 1] and shared across panels."""
    n_heads = in_cxt_attn.numel()
    n_neurons = in_cxt_mlp.numel()

    k_heads = np.arange(1, n_heads + 1)
    iou_heads = [iou_at_k(in_cxt_attn, recall_attn, int(k)) for k in k_heads]
    rand_heads = k_heads / (2 * n_heads - k_heads)

    k_neurons = np.unique(np.logspace(0, np.log10(n_neurons), 60).astype(int))
    iou_neurons = [iou_at_k(in_cxt_mlp, recall_mlp, int(k)) for k in k_neurons]
    rand_neurons = k_neurons / (2 * n_neurons - k_neurons)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].plot(k_heads, iou_heads, lw=1.8, label="observed")
    axes[0].plot(k_heads, rand_heads, "--", color="gray", lw=1, label="random")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("IoU(top-K)")
    axes[0].set_title(f"Attention heads (total = {n_heads})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_neurons, iou_neurons, lw=1.8, label="observed")
    axes[1].plot(k_neurons, rand_neurons, "--", color="gray", lw=1, label="random")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("K (log scale)")
    axes[1].set_title(f"MLP neurons (total = {n_neurons})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[0].set_ylim(0, 1)
    fig.suptitle("IoU of top-K components: in_cxt vs recall")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_neurons(in_cxt_mlp, recall_mlp, n: int, save_path: Path):
    """1x2 figure: scatter of top-N MLP neurons (in_cxt vs recall) and the
    distribution of those neurons across layers."""
    L, d_ff = in_cxt_mlp.shape
    diff = in_cxt_mlp - recall_mlp
    topn_idx = torch.abs(diff.flatten()).topk(n).indices
    layers = (topn_idx // d_ff).tolist()
    neurons = (topn_idx % d_ff).tolist()
    in_cxt_vals = [in_cxt_mlp[l, ne].item() for l, ne in zip(layers, neurons)]
    recall_vals = [recall_mlp[l, ne].item() for l, ne in zip(layers, neurons)]
    diff_vals = [in_cxt_vals[i] - recall_vals[i] for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(recall_vals, in_cxt_vals, c=layers, cmap="viridis",
                         s=40, alpha=0.85, edgecolors="k", linewidths=0.3)
    lim = max(max(map(abs, in_cxt_vals)), max(map(abs, recall_vals))) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], "k--", alpha=0.4, lw=1)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_xlabel("recall attribution")
    axes[0].set_ylabel("in_cxt attribution")
    axes[0].set_title(f"Top-{n} MLP neurons (color = layer)")
    cbar = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("layer")

    top5 = sorted(range(n), key=lambda i: abs(diff_vals[i]), reverse=True)[:5]
    for i in top5:
        axes[0].annotate(f"L{layers[i]}.N{neurons[i]}",
                         (recall_vals[i], in_cxt_vals[i]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)

    counts = np.zeros(L, dtype=int)
    for l in layers:
        counts[l] += 1
    axes[1].bar(np.arange(L), counts)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel(f"count of top-{n} neurons")
    axes[1].set_title("Layer distribution")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=64,
                        help="Top-K neurons per layer in MLP heatmap.")
    parser.add_argument("--topn", type=int, default=100,
                        help="Global top-N MLP neurons by |diff| in CSV.")
    parser.add_argument("--metric", choices=["log_prob", "logit_diff"],
                        default="log_prob",
                        help="Which attribution metric's outputs to plot.")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(exist_ok=True)
    in_cxt, recall = load_attr(args.metric)

    # Normalize by |mean metric| only for logit_diff, where the denominator
    # is bounded and the "fraction explained" framing is meaningful. For
    # log_prob the model is often near-saturated (mean log_prob ≈ 0), so the
    # ratio explodes and the raw log-prob units are more interpretable.
    if args.metric == "logit_diff":
        in_cxt_mlp = in_cxt["mlp"] / abs(in_cxt["metric"])
        in_cxt_attn = in_cxt["attn"] / abs(in_cxt["metric"])
        recall_mlp = recall["mlp"] / abs(recall["metric"])
        recall_attn = recall["attn"] / abs(recall["metric"])
    else:
        in_cxt_mlp = in_cxt["mlp"]
        in_cxt_attn = in_cxt["attn"]
        recall_mlp = recall["mlp"]
        recall_attn = recall["attn"]
    print(f"mean metric  in_cxt={in_cxt['metric']:.3f}  recall={recall['metric']:.3f}")

    plot_attn(in_cxt_attn, recall_attn,
              PLOTS_DIR / f"attn_heatmaps_{args.metric}.png")
    plot_mlp_topk(in_cxt_mlp, recall_mlp, args.topk,
                  PLOTS_DIR / f"mlp_topk_heatmaps_{args.metric}.png")
    plot_layer_summary(in_cxt_mlp, recall_mlp,
                       in_cxt_attn, recall_attn,
                       PLOTS_DIR / f"mlp_layer_summary_{args.metric}.png")
    write_top_neurons(in_cxt_mlp, recall_mlp, args.topn,
                      PLOTS_DIR / f"top_neurons_{args.metric}.csv")
    plot_top_neurons(in_cxt_mlp, recall_mlp, args.topn,
                     PLOTS_DIR / f"top_neurons_{args.metric}.png")
    plot_iou_curve(in_cxt_mlp, recall_mlp, in_cxt_attn, recall_attn,
                   PLOTS_DIR / f"iou_overlap_{args.metric}.png")

    print(f"Saved plots and CSV to {PLOTS_DIR}/  (metric={args.metric})")


if __name__ == "__main__":
    main()