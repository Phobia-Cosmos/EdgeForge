#!/usr/bin/env python3
"""Visualize complete 20-epoch EEG sequences and characteristic epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


FS_HZ = 100.0
EXPECTED_EPOCHS = 20
EXPECTED_CHANNELS = 8
EXPECTED_SAMPLES = 3000
CLASS_COLORS = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b", 3: "#e45756", 4: "#b279a2"}


def _sequence_files(root: Path, group: str, subject: int) -> list[Path]:
    return sorted((root / group / str(subject) / "data").glob("*.npy"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)


def _load_sequence(root: Path, group: str, subject: int, path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    labels = np.load(root / group / str(subject) / "label" / path.name, allow_pickle=False).reshape(-1).astype(np.int64, copy=False)
    if values.ndim != 3 or values.shape[1:] != (EXPECTED_CHANNELS, EXPECTED_SAMPLES):
        raise ValueError(f"expected (epoch, {EXPECTED_CHANNELS}, {EXPECTED_SAMPLES}), got {values.shape} for {path}")
    if len(values) != len(labels):
        raise ValueError(f"data/label length mismatch for {path}: {len(values)} != {len(labels)}")
    return values, labels


def _sequence_metrics(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    epoch_rms = np.sqrt(np.mean(np.square(values.astype(np.float64)), axis=(1, 2)))
    transition_count = int(np.sum(labels[1:] != labels[:-1]))
    median_index = int(np.argmin(np.abs(epoch_rms - np.median(epoch_rms))))
    first_transition = int(np.flatnonzero(labels[1:] != labels[:-1])[0] + 1) if transition_count else None
    return {
        "sequence_rms": float(np.sqrt(np.mean(np.square(values.astype(np.float64))))),
        "epoch_rms": epoch_rms.tolist(),
        "epoch_rms_cv": float(np.std(epoch_rms) / max(float(np.mean(epoch_rms)), 1e-30)),
        "label_transition_count": transition_count,
        "lowest_rms_epoch": int(np.argmin(epoch_rms)),
        "median_rms_epoch": median_index,
        "highest_rms_epoch": int(np.argmax(epoch_rms)),
        "first_label_transition_epoch": first_transition,
        "labels": labels.tolist(),
    }


def _choose_sequences(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose complementary sequences: label-rich, amplitude-variable, then typical."""
    if not entries or count <= 0:
        return []
    chosen: list[dict[str, Any]] = []
    criteria = [
        ("most label transitions", lambda item: (item["metrics"]["label_transition_count"], item["metrics"]["epoch_rms_cv"])),
        ("largest within-sequence RMS variation", lambda item: (item["metrics"]["epoch_rms_cv"], item["metrics"]["label_transition_count"])),
    ]
    for reason, key in criteria:
        candidates = [entry for entry in entries if entry not in chosen]
        if not candidates or len(chosen) >= count:
            break
        selected = max(candidates, key=key)
        selected = dict(selected)
        selected["selection_reason"] = reason
        chosen.append(selected)
    if len(chosen) < count:
        median_rms = float(np.median([entry["metrics"]["sequence_rms"] for entry in entries]))
        candidates = sorted(
            (entry for entry in entries if entry not in chosen),
            key=lambda item: abs(item["metrics"]["sequence_rms"] - median_rms),
        )
        for entry in candidates[: count - len(chosen)]:
            selected = dict(entry)
            selected["selection_reason"] = "typical sequence RMS"
            chosen.append(selected)
    return chosen


def _waveform_stack(values: np.ndarray, labels: np.ndarray, metrics: dict[str, Any], output: Path, subject: int, sequence: str, channel: int) -> None:
    n_epochs = len(values)
    time = np.arange(values.shape[-1]) / FS_HZ
    channel_values = values[:, channel].astype(np.float64, copy=False)
    scale = max(float(np.percentile(np.abs(channel_values), 99.5)), 1e-30)
    fig, (wave_ax, rms_ax) = plt.subplots(1, 2, figsize=(17, 12), gridspec_kw={"width_ratios": [4.8, 1.2]}, sharey=True)
    for epoch in range(n_epochs):
        trace = np.clip(channel_values[epoch] / scale, -1.3, 1.3) * 0.36 + epoch
        color = CLASS_COLORS.get(int(labels[epoch]), "#666666")
        wave_ax.plot(time, trace, color=color, linewidth=0.65)
    wave_ax.axhline(9.5, color="black", linestyle="--", linewidth=1.2)
    wave_ax.set_xlim(0.0, values.shape[-1] / FS_HZ)
    wave_ax.set_ylim(-0.7, n_epochs - 0.3)
    wave_ax.invert_yaxis()
    wave_ax.set_yticks(range(n_epochs), [f"E{epoch:02d} · class {int(labels[epoch])}" for epoch in range(n_epochs)])
    wave_ax.set_xlabel("time within each epoch (s)")
    wave_ax.set_ylabel("chronological epoch index")
    wave_ax.set_title(f"Channel {channel}: 20 complete 30-second waveforms\ncommon sequence scale = 99.5th percentile ({scale:.2e})")
    wave_ax.grid(axis="x", alpha=0.18)

    epoch_rms = np.asarray(metrics["epoch_rms"], dtype=np.float64)
    relative_rms = epoch_rms / max(float(np.median(epoch_rms)), 1e-30)
    rms_ax.plot(relative_rms, np.arange(n_epochs), color="#222222", marker="o", markersize=3, linewidth=1.0)
    rms_ax.axhline(9.5, color="black", linestyle="--", linewidth=1.2)
    markers = {
        "low": metrics["lowest_rms_epoch"],
        "median": metrics["median_rms_epoch"],
        "high": metrics["highest_rms_epoch"],
    }
    marker_colors = {"low": "#4c78a8", "median": "#54a24b", "high": "#e45756"}
    for name, epoch in markers.items():
        rms_ax.scatter([relative_rms[epoch]], [epoch], s=55, color=marker_colors[name], zorder=3, label=f"{name} RMS: E{epoch:02d}")
    if metrics["first_label_transition_epoch"] is not None:
        epoch = int(metrics["first_label_transition_epoch"])
        rms_ax.scatter([relative_rms[epoch]], [epoch], s=70, facecolors="none", edgecolors="#000000", linewidths=1.5, label=f"first class change: E{epoch:02d}")
    rms_ax.set_xlabel("epoch RMS / sequence median")
    rms_ax.set_title("Amplitude trajectory")
    rms_ax.grid(alpha=0.2)
    rms_ax.legend(loc="lower right", fontsize=8)

    class_legend = [Line2D([0], [0], color=color, lw=2, label=f"class {class_id}") for class_id, color in CLASS_COLORS.items()]
    wave_ax.legend(handles=class_legend, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.09), fontsize=8)
    fig.suptitle(f"Clean ISRUC medium · target subject {subject} · sequence file {sequence}.npy\ndashed boundary: E00--E09 adaptation, E10--E19 held-out evaluation", fontsize=14)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.90, wspace=0.12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _all_channel_heatmap(values: np.ndarray, labels: np.ndarray, output: Path, subject: int, sequence: str) -> None:
    scale = max(float(np.percentile(np.abs(values), 99.0)), 1e-30)
    normalized = np.clip(values.astype(np.float64) / scale, -1.0, 1.0)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True, sharey=True)
    image = None
    label_text = [f"E{epoch:02d}/C{int(label)}" for epoch, label in enumerate(labels)]
    for channel, ax in enumerate(axes.flat):
        image = ax.imshow(normalized[:, channel, :], aspect="auto", origin="upper", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                          extent=[0.0, values.shape[-1] / FS_HZ, len(values) - 0.5, -0.5])
        ax.axhline(9.5, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"channel {channel}")
        ax.set_xlabel("time within epoch (s)")
        ax.set_ylabel("epoch / class")
        ax.set_yticks(range(len(values)), label_text, fontsize=7)
    if image is not None:
        fig.subplots_adjust(left=0.10, right=0.88, bottom=0.06, top=0.91, hspace=0.35, wspace=0.18)
        colorbar_axis = fig.add_axes([0.91, 0.16, 0.015, 0.68])
        fig.colorbar(image, cax=colorbar_axis, label="amplitude / sequence-wide 99th percentile")
    fig.suptitle(f"Clean ISRUC medium · target subject {subject} · sequence {sequence}: all 20 epochs × 8 channels\ndashed boundary: adaptation/evaluation; row label E##/C# = epoch index/class code", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_sequences(data_root: str | Path, output_dir: str | Path, subjects: list[int], sequences_per_subject: int = 2, channel: int = 0) -> dict[str, Any]:
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selections: list[dict[str, Any]] = []
    for subject in subjects:
        entries = []
        for path in _sequence_files(root, "target", subject):
            values, labels = _load_sequence(root, "target", subject, path)
            if len(values) != EXPECTED_EPOCHS:
                continue
            entries.append({"subject": subject, "sequence": path.stem, "path": path, "values": values, "labels": labels, "metrics": _sequence_metrics(values, labels)})
        selections.extend(_choose_sequences(entries, sequences_per_subject))

    explanation = [
        "# Clean ISRUC medium: complete sequence waveform views",
        "",
        "每个 sequence 文件包含 20 个按时间排序的 30 秒 epoch；E00--E09 是 adaptation，E10--E19 是 held-out evaluation。这里读取原有 float32 数组，不在绘图脚本中重新滤波或重采样。睡眠标签仅写作 class 0--4，避免在没有核对上游映射时擅自赋予阶段名称。",
        "",
        "每个入选 sequence 有两张图：`waveform-stack` 使用同一个 sequence-wide 尺度叠放 channel 0 的 20 条完整波形，因此可以比较 epoch 间真实相对振幅；`all-channels` 用 8 个热图查看每个通道的 20×30 秒变化。颜色深浅不能跨不同图片直接比较，因为每张图片使用自己的 robust scale。",
        "",
        "选择规则优先保留睡眠标签转换最多的 sequence 和 epoch RMS 变化最大的 sequence。图中的 low/median/high RMS 只表示该 sequence 内的相对振幅，不自动等于坏数据或某种睡眠阶段。",
        "",
    ]
    serializable = []
    for item in selections:
        subject = int(item["subject"])
        sequence = str(item["sequence"])
        stem = f"subject-{subject}-sequence-{sequence}"
        waveform_name = f"{stem}-waveform-stack.png"
        channel_name = f"{stem}-all-channels.png"
        _waveform_stack(item["values"], item["labels"], item["metrics"], output / waveform_name, subject, sequence, channel)
        _all_channel_heatmap(item["values"], item["labels"], output / channel_name, subject, sequence)
        metrics = dict(item["metrics"])
        record = {"subject": subject, "sequence": sequence, "selection_reason": item["selection_reason"], "metrics": metrics, "figures": [waveform_name, channel_name]}
        serializable.append(record)
        explanation.extend([
            f"## Subject {subject}, sequence {sequence}",
            "",
            f"选择原因：{item['selection_reason']}。标签转换 {metrics['label_transition_count']} 次，epoch RMS 变异系数为 {metrics['epoch_rms_cv']:.3f}；低/中位/高 RMS 代表 epoch 分别是 E{metrics['lowest_rms_epoch']:02d}、E{metrics['median_rms_epoch']:02d}、E{metrics['highest_rms_epoch']:02d}。",
            "",
            f"- `{waveform_name}`：从上到下按 E00--E19 阅读。每条曲线都是 channel {channel} 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。",
            f"- `{channel_name}`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。",
            "",
        ])
    (output / "EXPLANATION.md").write_text("\n".join(explanation).rstrip() + "\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "analysis": "complete-eeg-sequence-visualization",
        "data_root": str(root),
        "sampling_rate_hz": FS_HZ,
        "epoch_samples": EXPECTED_SAMPLES,
        "epoch_duration_seconds": EXPECTED_SAMPLES / FS_HZ,
        "subjects": subjects,
        "sequences_per_subject": sequences_per_subject,
        "channel_for_waveform_stack": channel,
        "selections": serializable,
        "scientific_conclusion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--subjects", nargs="+", type=int, default=[2, 12, 14, 15])
    parser.add_argument("--sequences-per-subject", type=int, default=2)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.channel < EXPECTED_CHANNELS:
        parser.error(f"--channel must be in [0, {EXPECTED_CHANNELS - 1}]")
    summary = visualize_sequences(args.data_root, args.output_dir, args.subjects, args.sequences_per_subject, args.channel)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "selected_sequences": len(summary["selections"]), "output_dir": str(args.output_dir.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
