#!/usr/bin/env python3
"""Discover existing BrainUICL result files and write an EdgeForge catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeforge.raeeg_catalog import build_catalog, discover_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/undefined/Desktop/bci/code/tta_security/BrainUICL")
    parser.add_argument("--output", required=True, help="catalog JSON output path")
    parser.add_argument("--contains", help="only include result paths containing this substring")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-manifest-digest")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    paths = discover_results(root, pattern=args.contains, limit=args.limit)
    catalog = build_catalog(root, paths, dataset_manifest_digest=args.dataset_manifest_digest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "results": len(paths), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
