# Local EEG experiment data

EdgeForge does not ship real ISRUC/FACED EEG payloads. These are human
biomedical signals and remain under `/home/undefined/Disk/datasets/`. Create a
small local split with `scripts/create-eeg-mini-split.py`; the output contains
paired `.npy` files and SHA-256 provenance. Only synthetic fixtures and the
extraction code are suitable for source control until dataset license, ethics,
de-identification and public-release permission are explicitly confirmed.
