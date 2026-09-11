#!/usr/bin/env python3
"""Keep EEG image artifacts readable and archive machine-readable configs.

The operation is restricted to the generated experiment archive.  It never
touches the repository or source datasets, and writes a reversible move map.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _description(image: Path) -> str:
    name = image.name.lower()
    if "waveform" in name:
        return "波形图：比较个体或条件下的时域 EEG 形状与幅度。"
    if "spect" in name or "psd" in name:
        return "频谱/PSD 图：比较 0.5–40 Hz 频带能量分布；PSD 是功率谱密度。"
    if "rms" in name:
        return "RMS 图：显示每个 epoch 的整体信号强度轨迹。"
    if "pca" in name:
        return "PCA 图：把 RMS、频带功率等特征投影到低维，用于观察个体/条件分离。"
    if "distance" in name:
        return "个体距离图：显示基于标准化信号特征的个体间距离。"
    if "label" in name:
        return "标签先验图：显示睡眠阶段标签比例，不是模型激活证据。"
    if "polarity" in name:
        return "极性对照图：验证通道翻转后的波形镜像、PSD 保持和相关系数变化。"
    if "dose" in name or "lop" in name:
        return "LoP dose 曲线：展示不同条件/预算的 fresh-gap，必须结合严格 gate 解读。"
    return "实验图：用于描述该实验条件的 EEG 信号或结果，不能单独证明 LoP。"


def organize(root: Path, archive_name: str, archive_root: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    archive = (archive_root if archive_root is not None else root.parent / f"{root.name}-{archive_name}").resolve()
    if archive == root or root in archive.parents:
        raise ValueError("archive must be outside the visualization root")
    archive.mkdir(parents=True, exist_ok=True)
    image_dirs = sorted({path.parent for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and archive not in path.parents})
    moves: list[dict[str, str]] = []
    explanations: list[str] = []
    for directory in image_dirs:
        relative = directory.relative_to(root)
        moved_names: list[str] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() in IMAGE_SUFFIXES or path.name == "EXPLANATION.md":
                continue
            destination = archive / relative / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = archive / relative / f"{path.stem}.duplicate{path.suffix}"
            shutil.move(str(path), str(destination))
            moves.append({"from": str(path), "to": str(destination)})
            moved_names.append(path.name)
        images = sorted(path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        lines = [
            f"# {directory.name}",
            "",
            f"本目录只保留图片；机器可读文件已归档到 `{archive.name}/` 下的同一相对路径。",
            "这些图片是描述性 EEG/LoP 诊断，不能单独作为 LoP 结论；LoP 必须查看对应实验的 fresh-gap 严格 gate。",
            "",
            "## 图片说明",
            "",
        ]
        for image in images:
            lines.append(f"- `{image}`：{_description(Path(image))}")
        if moved_names:
            lines.extend(["", "归档文件：`" + "`、`".join(moved_names) + "`。"])
        (directory / "EXPLANATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        explanations.append(str(directory))

    # Archive every non-image file outside visualization directories.  The
    # result root is intentionally a human-facing image archive, so reports,
    # manifests and logs are kept in a sibling analysis directory.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or archive in path.parents or path.suffix.lower() in IMAGE_SUFFIXES:
            continue
        if path.name == "EXPLANATION.md" and any(path.parent == directory for directory in image_dirs):
            continue
        relative = path.relative_to(root)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = archive / relative.parent / f"{relative.stem}.duplicate{relative.suffix}"
        shutil.move(str(path), str(destination))
        moves.append({"from": str(path), "to": str(destination)})
    # Remove empty experiment/config directories left behind by the moves.
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
        if directory in image_dirs:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass
    index_lines = [
        "# EEG visualization archive",
        "",
        f"保留了 {len(image_dirs)} 个含图片目录。每个目录的 `EXPLANATION.md` 解释图片含义、坐标和阅读方式。",
        f"JSON、报告、日志和其它分析文件位于同级归档：`{archive}`。移动清单为 `{archive / 'organization-manifest.json'}`。",
        "",
        "## 可视化目录",
        "",
    ]
    for directory in image_dirs:
        index_lines.append(f"- `{directory.relative_to(root)}`：查看该目录的 `EXPLANATION.md`。")
    (root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "edgeforge.eeg-visual-artifact-organization.v1",
        "root": str(root),
        "archive": str(archive),
        "image_directories": explanations,
        "move_count": len(moves),
        "moves": moves,
    }
    (archive / "organization-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-name", default="analysis-configs-20260911")
    parser.add_argument("--archive-root", type=Path, default=None, help="sibling directory for reports/configs; must be outside --root")
    args = parser.parse_args()
    manifest = organize(args.root, args.archive_name, args.archive_root)
    print(json.dumps({"status": "ok", "image_directories": len(manifest["image_directories"]), "move_count": manifest["move_count"], "archive": manifest["archive"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
