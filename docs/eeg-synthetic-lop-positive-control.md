# Synthetic LoP positive control

EdgeForge includes a clearly labeled synthetic EEG positive control for validating the fresh-vs-warm LoP gate. It is not a substitute for ISRUC or a scientific claim. The generated data use the same EEG-shaped epoch contract `(8,3000)`, while labels are controlled by 16 latent channel/half-window features.

The source task uses latent axis 0. The ordered target stream uses orthogonal axes `(15,1,14,2,13,3,12,4)`. The positive-control arm pretrains a warm model, then sets the encoder gradient scale to zero before sequential adaptation. Fresh models remain fully trainable. The no-lesion arm uses the same generated data and protocol without setting the gradient scale to zero.

Generate data, run both arms, and audit the result:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/generate-synthetic-lop-positive-control.py \
  --output /home/undefined/UbuntuData/datasets/edgeforge-synthetic-eeg-lop-positive-control-v1

PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-synthetic-lop-positive-control.py \
  --data-root /home/undefined/UbuntuData/datasets/edgeforge-synthetic-eeg-lop-positive-control-v1 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/synthetic-positive-control-v1

PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/audit-eeg-continuous-lop.py \
  --summary /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/synthetic-positive-control-v1/summary.json \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/synthetic-positive-control-v1/audit-v1 \
  --budget 50
```

In the current run, the positive-control arm is `candidate` at budgets 25 and 50: all 8 transitions and all 3 seeds have positive fresh-gap with positive bootstrap lower bounds. The no-lesion arm remains `blocked-inconsistent-direction`. This is evidence that the EdgeForge gate can detect an injected plasticity impairment; it is not evidence that naturally occurring LoP has been established.
