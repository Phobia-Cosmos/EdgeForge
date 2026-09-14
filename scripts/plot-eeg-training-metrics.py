#!/usr/bin/env python3
"""Plot EEG LoP training diagnostics from the structured summary JSON.

The plots are descriptive. They do not turn rank or gradient changes into a
LoP claim; the fixed-budget fresh-vs-warm gap remains the outcome gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"warm": "#d95f02", "fresh": "#1b9e77"}
LABELS = {"warm": "warm checkpoint", "fresh": "fresh initialization"}


def _mean_std(rows: list[dict[str, Any]], arm: str, budget: int, key: str) -> tuple[float, float, int]:
    values = [float(row[key]) for row in rows if row["arm"] == arm and int(row["budget"]) == int(budget)]
    if not values:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0), len(values)


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("adaptation budget (steps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)


def _plot_budget_metrics(summary: dict[str, Any], output: Path) -> None:
    rows = list(summary["rows"])
    budgets = [int(x) for x in summary["budgets"]]
    specs = [
        ("embedding_effective_rank", "Embedding effective rank", "effective rank"),
        ("embedding_effective_rank_normalized", "Normalized embedding rank", "ER / min(N,D)"),
        ("block1_effective_rank", "Block-1 effective rank", "effective rank"),
        ("gradient_norm_l2", "Gradient L2 norm", "L2 norm (log scale)"),
        ("gradient_nonzero_fraction", "Gradient nonzero fraction", "fraction"),
        ("parameter_global_relative_update", "Parameter relative update", "relative update"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.flat, specs):
        for arm in ("warm", "fresh"):
            means, stds = [], []
            xs = []
            for budget in budgets:
                mean, std, count = _mean_std(rows, arm, budget, key)
                if count:
                    xs.append(budget)
                    means.append(mean)
                    stds.append(std)
            ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, capsize=3,
                        color=COLORS[arm], label=LABELS[arm])
        _style_axis(ax, title, ylabel)
        if key == "gradient_norm_l2":
            ax.set_yscale("log")
        if key == "gradient_nonzero_fraction":
            ax.set_ylim(0.45, 1.03)
        if key == "parameter_global_relative_update":
            ax.set_ylim(bottom=-0.005)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "TCN EEG diagnostics across adaptation budgets\n"
        "mean ± SD over 8 target stages × 3 seeds; descriptive, not a LoP gate",
        fontsize=14,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_fresh_gap(summary: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = [row for row in summary["rows"] if row["arm"] == "warm"]
    budgets = [int(x) for x in summary["budgets"]]
    means, stds, positive, counts = [], [], [], []
    for budget in budgets:
        values = [float(row["fresh_gap"]) for row in rows if int(row["budget"]) == budget]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1) if len(values) > 1 else 0.0))
        positive.append(float(np.mean(np.asarray(values) > 0.0)))
        counts.append(len(values))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.errorbar(budgets, means, yerr=stds, color="#4c78a8", marker="o", linewidth=2,
                capsize=4, label="fresh-gap mean ± SD")
    ax.scatter(budgets, means, color="#4c78a8", s=45, zorder=3)
    for x, y, p in zip(budgets, means, positive):
        ax.annotate(f"positive={p:.0%}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#345b84")
    ax.set_xlabel("adaptation budget (steps)")
    ax.set_ylabel("fresh-gap = accuracy_fresh − accuracy_warm")
    ax.set_title("TCN fixed-budget plasticity outcome\n8 transitions × 3 seeds per budget")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "budgets": budgets,
        "fresh_gap_mean": means,
        "fresh_gap_std": stds,
        "positive_fraction": positive,
        "row_count": counts,
    }


def _plot_associations(summary: dict[str, Any], output: Path) -> dict[str, float]:
    rows = [row for row in summary["rows"] if row["arm"] == "warm"]
    pairs = [
        ("embedding_effective_rank", "embedding ER", "ER"),
        ("block1_effective_rank", "block1 ER", "block1 ER"),
        ("gradient_nonzero_fraction", "gradient nonzero", "gradient NZ"),
        ("gradient_norm_l2", "gradient norm", "gradient norm"),
    ]
    gaps = np.asarray([float(row["fresh_gap"]) for row in rows], dtype=float)
    budgets = np.asarray([int(row["budget"]) for row in rows], dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    correlations: dict[str, float] = {}
    for ax, (key, title, short) in zip(axes.flat, pairs):
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        corr = float(np.corrcoef(values, gaps)[0, 1]) if np.std(values) > 0 and np.std(gaps) > 0 else float("nan")
        correlations[key] = corr
        scatter = ax.scatter(values, gaps, c=budgets, cmap="viridis", s=38, alpha=0.8,
                             edgecolors="white", linewidths=0.25)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xlabel(short)
        ax.set_ylabel("fresh-gap")
        ax.set_title(f"{title} vs fresh-gap\nPearson r = {corr:.3f}")
        ax.grid(alpha=0.20)
        if key == "gradient_norm_l2":
            ax.set_xscale("log")
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("adaptation budget")
    fig.suptitle("Mechanism candidates versus LoP outcome\nwarm arm, 24 stage-seed cells per budget aggregate", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return correlations


def _plot_stage_heatmaps(summary: dict[str, Any], output: Path) -> None:
    rows = [row for row in summary["rows"] if row["arm"] == "warm"]
    budgets = [int(x) for x in summary["budgets"]]
    stages = sorted({int(row["stage"]) for row in rows})
    stage_subject = {int(row["stage"]): int(row["subject"]) for row in rows}
    specs = [
        ("embedding_effective_rank", "warm embedding ER", "viridis"),
        ("block1_effective_rank", "warm block1 ER", "viridis"),
        ("gradient_nonzero_fraction", "warm gradient nonzero", "magma"),
        ("gradient_norm_l2", "warm gradient norm", "magma"),
        ("fresh_gap", "fresh-gap", "RdBu_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes.flat, specs):
        matrix = np.full((len(stages), len(budgets)), np.nan, dtype=float)
        for i, stage in enumerate(stages):
            for j, budget in enumerate(budgets):
                values = [float(row[key]) for row in rows if int(row["stage"]) == stage and int(row["budget"]) == budget]
                if values:
                    matrix[i, j] = float(np.mean(values))
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(len(budgets)), [str(x) for x in budgets])
        ax.set_yticks(range(len(stages)), [f"S{stage} / subj {stage_subject[stage]}" for stage in stages], fontsize=8)
        ax.set_xlabel("adaptation budget")
        ax.set_ylabel("target stage")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, shrink=0.82)
    axes.flat[-1].axis("off")
    fig.suptitle("Warm diagnostics by target stage and adaptation budget\n"
                 "cell values are mean over 3 seeds; use this to find stage-local anomalies", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_explanation(output: Path, summary: dict[str, Any], gap: dict[str, Any], correlations: dict[str, float]) -> None:
    text = f"""# EEG 训练指标可视化

本目录中的图来自 `rms-equalized-tcn-source-epochs10-diagnostics-v1` 的结构化诊断结果，共 {summary['row_count']} 行：8 个 target stage、3 个 seed、fresh/warm 两个 arm 和 5 个 adaptation budget。误差线是跨 stage-seed cell 的标准差。

## 图 1：`training-metrics-by-budget.png`

六个面板分别显示 embedding effective rank、normalized effective rank、首个卷积块 effective rank、梯度 L2 范数、梯度非零比例和参数相对更新量。warm 是跨 stage 传递的 checkpoint，fresh 是每个 target stage 重新初始化的模型。梯度范数使用对数纵轴，因为不同 cell 的数值跨度较大。图中只能看预算相关的即时变化，不能把横轴 budget 当成长期 stream age；要判断论文意义上的“训练越久越失去可塑性”，还需要沿 stream age 保存连续 checkpoint。

当前最重要的读法是：warm embedding ER 没有单调下降，先降后升再回落；gradient nonzero fraction 约保持在 0.58–0.60；classifier-input near-zero fraction 在原始表格中所有 cell 都为 0（因此没有额外画成一条完全重合的零线）。这不支持“整体谱坍缩或全网络失活”的结论。

## 图 2：`fresh-gap-by-budget.png`

纵轴是 `fresh-gap = accuracy_fresh − accuracy_warm`，0 线以上表示 fresh 在固定预算后更好，0 线以下表示 warm 更好。误差线显示 24 个 stage-seed cell 的离散程度，文字标注是 fresh-gap 为正的 cell 比例。它是 LoP 的主要结果图，但均值为正仍不够；严格 gate 还要求跨 seed、transition 的方向一致并且不确定性下界为正。

## 图 3：`metric-vs-fresh-gap.png`

每个点是一个 warm stage-seed cell，颜色表示 adaptation budget。它用来检查 rank、梯度可达性或梯度范数是否与 fresh-gap 同步变化；散点混合或相关系数接近 0 时，不能把该指标称为 LoP 机制。当前 warm 初始 embedding ER 与最终 fresh-gap 的探索性相关约为 -0.112；本图的预算级 cell 相关如下：

"""
    for key, value in correlations.items():
        text += f"- `{key}` vs fresh-gap：Pearson r = {value:.3f}。\n"
    text += f"""
## 图 4：`stage-heatmaps.png`

每个热图的纵轴是连续 target stage（同时标出 subject），横轴是 adaptation budget；每个格子是 3 个 seed 的均值。它用于发现被总体均值掩盖的局部异常，例如某个 subject 的 rank 突然降低或某一 stage 的 fresh-gap 方向相反。热图仍然是描述性结果，不能把单个异常格子直接解释为 LoP。

## 当前结果的结论

图中可以直观看到 rank、谱代理和梯度指标如何随 adaptation budget 变化，但它们不是独立的 LoP 证据。当前连续 medium 的 budget=50 最终 fresh-gap 均值约为 -0.008（TCN），且严格 gate 未通过；因此这些图支持“transfer sensitivity/预算依赖变化”的描述，不支持“ISRUC 已出现自然 LoP”。详细数值表见 [training-metrics-summary-20260912.md](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop-analysis-20260911/training-metrics-summary-20260912.md)。

注意：这里的 rank 是 calibration 表征矩阵的 effective rank，`sigma_max` 也只是激活表征的最大奇异值代理，不是完整参数奇异值谱。后续若要绘制真正的参数谱曲线，应在每个 checkpoint 保存完整 singular-value summary，并固定矩阵展平方式、层名和 calibration manifest。
"""
    (output / "EXPLANATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-summary", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_budget_metrics(summary, args.output_dir / "training-metrics-by-budget.png")
    gap = _plot_fresh_gap(summary, args.output_dir / "fresh-gap-by-budget.png")
    correlations = _plot_associations(summary, args.output_dir / "metric-vs-fresh-gap.png")
    _plot_stage_heatmaps(summary, args.output_dir / "stage-heatmaps.png")
    _write_explanation(args.output_dir, summary, gap, correlations)
    generated = {
        "schema_version": 1,
        "source_summary": str(args.summary_json.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "figures": [
            "training-metrics-by-budget.png",
            "fresh-gap-by-budget.png",
            "metric-vs-fresh-gap.png",
            "stage-heatmaps.png",
        ],
        "fresh_gap": gap,
        "correlations": correlations,
        "scientific_conclusion_allowed": False,
    }
    if args.archive_summary:
        args.archive_summary.parent.mkdir(parents=True, exist_ok=True)
        args.archive_summary.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(generated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
