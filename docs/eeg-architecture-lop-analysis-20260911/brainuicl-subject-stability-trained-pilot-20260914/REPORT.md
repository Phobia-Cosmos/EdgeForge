# BrainUICL subject stability audit

This run processed 100 sequences, 2000 epoch tokens and 20 subjects.

The model was trained for 5 epochs on medium source subjects in this run. This is a small supervised pilot, not a full-data BrainUICL checkpoint.

Stages are: EEG branch pooled 512-D, EOG branch pooled 512-D, fusion 512-D, Transformer repeated layers 1-3 at 512-D, classifier input 128-D, and 5-D logits. Epoch probes use all 20 tokens but hold out complete sequences; sequence probes use one median vector per sequence.

Read `brainuicl-stage-pca.png` for sequence-level individual differences and `brainuicl-stage-stability.png` for identity accuracy and between/within geometry. A trained-checkpoint result must be rerun with the same manifest and compared stage by stage.
