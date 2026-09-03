EdgeForge 0.17.0 local evidence archive.
The ISRUC mini split and benchmark outputs remain on shared storage and are
not committed because the payload is human EEG. This repository archive keeps
the version metadata, commands and interpretation boundary only.

Create the split:
PYTHONPATH=src python3 scripts/create-eeg-mini-split.py --source-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 --output-root /home/undefined/Disk/datasets/edgeforge-eeg-mini/v0.17.0

Run benchmark:
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python scripts/benchmark-eeg-architectures.py --data-root /home/undefined/Disk/datasets/edgeforge-eeg-mini/v0.17.0 --output-root /home/undefined/Disk/ai-storage/EdgeForge/eeg-architecture/v0.17.0 --architectures lop_mlp eegnet tcn transformer brainuicl --seeds 4321 4322 4323 --epochs 1 --adapt-steps 1 --batch-size 8 --latency-repeats 2

Validation: 15/15 CPU smoke runs succeeded; scientific_conclusion_allowed=false.
