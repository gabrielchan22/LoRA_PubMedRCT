"""
Accuracy-vs-rank and memory-vs-rank plots from results.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_vs_rank(results_csv: str = "results.csv", out_path: str = "accuracy_vs_rank.png"):
    df = pd.read_csv(results_csv)
    lora_df = df[df["method"] == "lora"]
    baseline_acc = df[df["method"] == "full_finetune"]["accuracy"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, group in lora_df.groupby("alpha_mode"):
        group = group.sort_values("rank")
        ax.plot(group["rank"], group["accuracy"], marker="o", label=mode)
    ax.axhline(baseline_acc, linestyle="--", color="gray", label="full fine-tune")
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Test accuracy")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.set_title("Accuracy vs LoRA rank")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_memory_vs_rank(results_csv: str = "results.csv", out_path: str = "memory_vs_rank.png"):
    df = pd.read_csv(results_csv)
    lora_df = df[df["method"] == "lora"]
    baseline_mem = df[df["method"] == "full_finetune"]["peak_memory_mb"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, group in lora_df.groupby("alpha_mode"):
        group = group.sort_values("rank")
        ax.plot(group["rank"], group["peak_memory_mb"], marker="o", label=mode)
    ax.axhline(baseline_mem, linestyle="--", color="gray", label="full fine-tune")
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.set_title("Peak memory vs LoRA rank")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


if __name__ == "__main__":
    plot_accuracy_vs_rank()
    plot_memory_vs_rank()
