#!/usr/bin/env bash
set -euo pipefail

ARCHITECTURE="${1:?usage: run-full-isruc-gpu.sh ARCHITECTURE OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?usage: run-full-isruc-gpu.sh ARCHITECTURE OUTPUT_ROOT}"

BASE="/public/home/u43077/lzh/ai/projects/EdgeForge"
DATA_ROOT="/public/home/u43077/lzh/ai/data/isruc-full-groups-v1"
PYTHON="/public/home/u43077/lzh/python-envs/llm-py311/bin/python"

cd "$BASE"
export PYTHONPATH="$BASE/src"

SOURCE_SUBJECTS=(1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60)
TARGET_SUBJECTS=(61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80)
RETENTION_SUBJECTS=(81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100)

exec "$PYTHON" -u scripts/run-eeg-architecture-continuous-lop.py \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --architectures "$ARCHITECTURE" \
  --seeds 4321 4322 4323 \
  --source-subjects "${SOURCE_SUBJECTS[@]}" \
  --target-subjects "${TARGET_SUBJECTS[@]}" \
  --retention-subjects "${RETENTION_SUBJECTS[@]}" \
  --budgets 0 5 10 25 50 \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-3 \
  --adapt-lr 1e-3 \
  --device cuda \
  --adaptation-strategy plain
