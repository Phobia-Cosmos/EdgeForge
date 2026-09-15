#!/usr/bin/env python3
"""Audit subject information across BrainUICL representation stages.

The default run is explicitly a random-initialization architecture audit when
native BrainUICL parameter checkpoints are unavailable.  It reads processed
ISRUC sequences, extracts the two convolution branches, fusion, each repeated
Transformer layer, classifier input and logits, then evaluates subject
separability with a sequence-disjoint split.  It never modifies the external
BrainUICL tree or the dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STAGES = (
    "eeg_branch",
    "eog_branch",
    "fusion",
    "transformer_layer1",
    "transformer_layer2",
    "transformer_layer3",
    "classifier_input",
    "logits",
)


def sequence_files(root: Path) -> list[tuple[int, str, Path, Path]]:
    rows = []
    for split in ("source", "target", "retention"):
        split_root = root / split
        if not split_root.is_dir():
            continue
        for subject_root in sorted((p for p in split_root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            for data_path in sorted((subject_root / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                label_path = subject_root / "label" / data_path.name
                if label_path.is_file():
                    rows.append((int(subject_root.name), f"{split}:{subject_root.name}:{data_path.stem}", data_path, label_path))
    return rows


def split_by_sequence(subjects: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(len(subjects), dtype=bool)
    test = np.zeros(len(subjects), dtype=bool)
    for sid in np.unique(subjects):
        indices = np.flatnonzero(subjects == sid)
        ordered = list(dict.fromkeys(str(groups[i]) for i in indices))
        cutoff = max(1, len(ordered) // 2)
        train_groups = set(ordered[:cutoff])
        train[indices] = np.asarray([str(groups[i]) in train_groups for i in indices])
        test[indices] = ~train[indices]
    return train, test


def probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    train, test = split_by_sequence(y, groups)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    model.fit(x[train], y[train])
    prediction = model.predict(x[test])
    return {
        "accuracy": float(accuracy_score(y[test], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
        "train_epochs": int(train.sum()),
        "test_epochs": int(test.sum()),
    }


def sequence_profile_stability(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Compare each subject's first/second sequence profile in standardized space."""
    z = StandardScaler().fit_transform(x)
    correlations = []
    distances = []
    within = []
    between = []
    sequence_keys = np.asarray(groups, dtype=object)
    for sid in np.unique(y):
        keys = list(dict.fromkeys(str(g) for g in sequence_keys[y == sid]))
        if len(keys) < 2:
            continue
        first = []
        second = []
        for key in keys[: len(keys) // 2]:
            first.append(np.median(z[sequence_keys == key], axis=0))
        for key in keys[len(keys) // 2 :]:
            second.append(np.median(z[sequence_keys == key], axis=0))
        profile_first = np.median(np.asarray(first), axis=0)
        profile_second = np.median(np.asarray(second), axis=0)
        correlations.append(float(np.corrcoef(profile_first, profile_second)[0, 1]))
        distances.append(float(np.linalg.norm(profile_first - profile_second)))
        centroid = np.median(z[y == sid], axis=0)
        within.extend(np.linalg.norm(z[y == sid] - centroid[None, :], axis=1).tolist())
    centroids = [np.median(z[y == sid], axis=0) for sid in np.unique(y)]
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return {
        "profile_first_second_correlation_mean": float(np.nanmean(correlations)),
        "profile_first_second_correlation_median": float(np.nanmedian(correlations)),
        "profile_first_second_distance_mean": float(np.mean(distances)),
        "within_subject_epoch_radius_median": float(np.median(within)),
        "between_subject_centroid_distance_median": float(np.median(between)),
        "between_within_ratio": float(np.median(between) / max(float(np.median(within)), 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, str(args.brainuicl_root.resolve()))
    import torch
    from model.pretrain_net import FeatureExtractor

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cpu")
    model_args = SimpleNamespace(dataset="ISRUC", device=device)
    frontend = FeatureExtractor(model_args).eval().to(device)
    from model.pretrain_net import SleepMLP, TransformerEncoder

    transformer = TransformerEncoder(model_args).eval().to(device)
    head = SleepMLP(model_args).eval().to(device)
    rows = sequence_files(args.data_root.resolve())
    if args.max_sequences > 0:
        rows = rows[: args.max_sequences]
    if not rows:
        raise RuntimeError(f"no processed ISRUC sequences under {args.data_root}")

    training_history = []
    if args.train_epochs > 0:
        train_rows = [row for row in rows if row[1].startswith("source:")]
        if not train_rows:
            raise RuntimeError("--train-epochs requires a data root with source subjects")
        parameters = list(frontend.parameters()) + list(transformer.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, betas=(0.5, 0.99), weight_decay=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch_index in range(args.train_epochs):
            frontend.train()
            transformer.train()
            head.train()
            generator = np.random.RandomState(args.seed + epoch_index)
            order = generator.permutation(len(train_rows))
            losses = []
            correct = 0
            count = 0
            for start in range(0, len(order), max(1, args.train_batch_size)):
                batch_rows = [train_rows[int(i)] for i in order[start : start + args.train_batch_size]]
                batch_values = np.stack([np.load(row[2], allow_pickle=False).astype("float32", copy=False) for row in batch_rows])
                batch_labels = np.stack([np.load(row[3], allow_pickle=False).reshape(20).astype("int64", copy=False) for row in batch_rows])
                values = torch.from_numpy(batch_values).to(device)
                labels = torch.from_numpy(batch_labels).to(device)
                flat = values.reshape(-1, 8, 3000)
                fused = frontend(flat[:, 2:], flat[:, :2])
                encoded = transformer(fused)
                logits = head(encoded)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                prediction = logits.argmax(dim=1)
                correct += int((prediction == labels).sum())
                count += int(labels.numel())
            row = {"epoch": epoch_index + 1, "train_loss": float(np.mean(losses)), "train_accuracy": correct / max(count, 1)}
            training_history.append(row)
            print(json.dumps({"event": "train_epoch", **row}), flush=True)

    collected: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGES}
    subjects: list[np.ndarray] = []
    groups: list[str] = []
    sequence_meta = []
    with torch.inference_mode():
        for index, (sid, group, data_path, label_path) in enumerate(rows, start=1):
            values = np.load(data_path, allow_pickle=False).astype("float32", copy=False)
            labels = np.load(label_path, allow_pickle=False).reshape(-1)
            if values.shape != (20, 8, 3000) or labels.shape != (20,):
                raise ValueError(f"expected ISRUC (20,8,3000)/(20,), got {data_path}: {values.shape}/{labels.shape}")
            flat = torch.from_numpy(values).to(device)
            eeg, eog = flat[:, 2:], flat[:, :2]
            eeg_conv = frontend.FEBlock_EEG(eeg)
            eog_conv = frontend.FEBlock_EOG(eog)
            eeg_pooled = frontend.avg(eeg_conv).reshape(1, 20, 512)
            eog_pooled = frontend.avg(eog_conv).reshape(1, 20, 512)
            fused = frontend.fusion(torch.cat((eeg_pooled.reshape(20, 1, 512), eog_pooled.reshape(20, 1, 512)), dim=2)).reshape(1, 20, 512)
            encoded = fused
            layer_values = []
            for _ in range(3):
                encoded = transformer.encoder.encoder(encoded)
                layer_values.append(encoded)
            classifier_input = head.sleep_stage_mlp(encoded)
            logits = head.sleep_stage_classifier(classifier_input).permute(0, 2, 1)
            values_by_stage = {
                "eeg_branch": eeg_pooled[0].cpu().numpy(),
                "eog_branch": eog_pooled[0].cpu().numpy(),
                "fusion": fused[0].cpu().numpy(),
                "transformer_layer1": layer_values[0][0].cpu().numpy(),
                "transformer_layer2": layer_values[1][0].cpu().numpy(),
                "transformer_layer3": layer_values[2][0].cpu().numpy(),
                "classifier_input": classifier_input[0].cpu().numpy(),
                # Native SleepMLP returns [B, classes, T]; the audit uses
                # one row per epoch token, so restore [T, classes].
                "logits": logits[0].transpose(0, 1).cpu().numpy(),
            }
            for stage in STAGES:
                collected[stage].append(values_by_stage[stage])
            subjects.append(np.full(20, sid, dtype=np.int64))
            groups.extend([group] * 20)
            sequence_meta.append({"subject": sid, "group": group, "data": str(data_path), "labels": labels.astype(int).tolist()})
            if index % 10 == 0 or index == len(rows):
                print(json.dumps({"event": "sequence_done", "index": index, "total": len(rows), "subject": sid}), flush=True)

    y = np.concatenate(subjects)
    g = np.asarray(groups, dtype=object)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "subjects.npy", y)
    np.save(output / "groups.npy", g)
    (output / "sequence-manifest.json").write_text(json.dumps(sequence_meta, indent=2) + "\n", encoding="utf-8")

    result = {
        "schema_version": 1,
        "dataset": "ISRUC",
        "data_root": str(args.data_root.resolve()),
        "brainuicl_root": str(args.brainuicl_root.resolve()),
        "initialization": "medium-source-supervised" if args.train_epochs > 0 else "random",
        "checkpoint_status": "trained in this run on medium source subjects; not a full-data checkpoint" if args.train_epochs > 0 else "unavailable; this is an architecture audit, not a trained-model claim",
        "seed": args.seed,
        "training": {"epochs": args.train_epochs, "batch_size_sequences": args.train_batch_size, "learning_rate": args.learning_rate, "source_sequence_count": len([row for row in rows if row[1].startswith("source:")]), "history": training_history},
        "sequence_count": len(rows),
        "epoch_count": int(len(y)),
        "subject_count": int(len(np.unique(y))),
        "input_layout": "B,T,C,S = (sequence,20,8,3000)",
        "stage_shapes": {"eeg_branch": [512], "eog_branch": [512], "fusion": [512], "transformer_layer1": [512], "transformer_layer2": [512], "transformer_layer3": [512], "classifier_input": [128], "logits": [5]},
        "stages": {},
        "scientific_conclusion_allowed": False,
    }
    for stage in STAGES:
        matrix = np.concatenate(collected[stage], axis=0).astype(np.float32)
        np.save(output / f"{stage}-epoch-embeddings.npy", matrix)
        # One sequence-level point is the median of its 20 epoch tokens.
        sequence_matrix = np.asarray([np.median(arr, axis=0) for arr in collected[stage]], dtype=np.float32)
        np.save(output / f"{stage}-sequence-embeddings.npy", sequence_matrix)
        stage_probe = probe(matrix, y, g)
        sequence_subjects = np.asarray([item[0] for item in [(m["subject"], m["group"]) for m in sequence_meta]], dtype=np.int64)
        sequence_groups = np.asarray([m["group"] for m in sequence_meta], dtype=object)
        sequence_probe = probe(sequence_matrix, sequence_subjects, sequence_groups)
        stability = sequence_profile_stability(matrix, y, g)
        raw_sequence_std = float(np.mean(np.std(sequence_matrix.astype(np.float64), axis=0)))
        raw_sequence_norm = float(np.mean(np.linalg.norm(sequence_matrix.astype(np.float64), axis=1)))
        result["stages"][stage] = {"epoch_probe": stage_probe, "sequence_probe": sequence_probe, "stability": stability, "raw_sequence_coordinate_std_mean": raw_sequence_std, "raw_sequence_norm_mean": raw_sequence_norm, "raw_relative_variation": raw_sequence_std / max(raw_sequence_norm, 1e-30)}

    (output / "brainuicl-subject-stability-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # PCA visualization uses sequence-level medians to avoid drawing 2,000
    # overlapping epoch dots while retaining the epoch-level probe above.
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    subject_sequence = np.asarray([m["subject"] for m in sequence_meta], dtype=np.int64)
    for ax, stage in zip(axes.flat, STAGES):
        seq = np.load(output / f"{stage}-sequence-embeddings.npy")
        projection = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(seq))
        scatter = ax.scatter(projection[:, 0], projection[:, 1], c=subject_sequence, cmap="turbo", s=28, alpha=0.85)
        ax.set_title(stage)
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.grid(alpha=0.18)
    axes.flat[-1].axis("off")
    run_label = "medium-source-supervised" if args.train_epochs > 0 else "random-init"
    fig.suptitle(f"BrainUICL {run_label} subject stability audit: sequence medians")
    fig.tight_layout()
    fig.savefig(output / "brainuicl-stage-pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    labels = list(STAGES)
    epoch_acc = [result["stages"][s]["epoch_probe"]["accuracy"] for s in labels]
    seq_acc = [result["stages"][s]["sequence_probe"]["accuracy"] for s in labels]
    ratios = [result["stages"][s]["stability"]["between_within_ratio"] for s in labels]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, epoch_acc, width=0.36, label="epoch probe")
    axes[0].bar(x + 0.18, seq_acc, width=0.36, label="sequence median probe")
    axes[0].axhline(1.0 / len(np.unique(y)), color="black", linestyle="--", linewidth=1, label="chance")
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("held-out subject accuracy")
    axes[0].set_title("Identity separability by stage")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, ratios, color="tab:orange")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("between-subject / within-subject distance")
    axes[1].set_title("Subject stability geometry")
    fig.tight_layout()
    fig.savefig(output / "brainuicl-stage-stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# BrainUICL subject stability audit",
        "",
        f"This run processed {len(rows)} sequences, {len(y)} epoch tokens and {len(np.unique(y))} subjects.",
        "",
        (f"The model was trained for {args.train_epochs} epochs on medium source subjects in this run. This is a small supervised pilot, not a full-data BrainUICL checkpoint." if args.train_epochs > 0 else "The model was random-initialized because no native BrainUICL parameter checkpoint was present. Therefore the plots measure what the architecture preserves from the input before training, not what a trained BrainUICL model has learned."),
        "",
        "Stages are: EEG branch pooled 512-D, EOG branch pooled 512-D, fusion 512-D, Transformer repeated layers 1-3 at 512-D, classifier input 128-D, and 5-D logits. Epoch probes use all 20 tokens but hold out complete sequences; sequence probes use one median vector per sequence.",
        "",
        "Read `brainuicl-stage-pca.png` for sequence-level individual differences and `brainuicl-stage-stability.png` for identity accuracy and between/within geometry. A trained-checkpoint result must be rerun with the same manifest and compared stage by stage.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "EXPLANATION.md").write_text("\n".join([
        "# 图表说明",
        "",
        "`brainuicl-stage-pca.png` 每个点是一个 sequence 的 20-epoch 中位 embedding，不是单个 epoch；颜色是 subject。横向 spread 表示不同个体/sequence 的表征差异，纵向同色散开表示被试内状态或采集漂移。PCA 只用于可视化，不是训练目标。",
        "",
        "`brainuicl-stage-stability.png` 左图比较 epoch token 与 sequence median 的身份 probe。sequence median 更接近个体画像，epoch probe 更容易受睡眠阶段影响。右图是标准化空间中的 between-subject centroid distance / within-subject radius；大于 1 才表示被试间中心差异超过被试内波动。",
        "",
        (f"本次在 medium source subjects 上监督训练 {args.train_epochs} 个 epoch；它可以描述小样本训练后的分层表征，但不能替代完整 98 人和正式 BrainUICL checkpoint。" if args.train_epochs > 0 else "本次是 random-init architecture audit。若某层准确率较高，只说明随机卷积/Transformer 仍保留了输入幅度、频谱和通道结构，不能说模型已经学会个体身份。训练 checkpoint 到位后必须重跑并报告差值。"),
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "sequences": len(rows), "epochs": len(y), "subjects": len(np.unique(y)), "checkpoint_status": result["checkpoint_status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
