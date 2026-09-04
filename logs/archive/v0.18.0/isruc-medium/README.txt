EdgeForge v0.18.0 ISRUC medium subset evidence.

The public payload contains 20 disjoint subjects, 100 paired sequence files
and 2,000 epochs. All 200 .npy files match the source/output SHA-256 values in
manifest.json. Ten unittest/model checks passed. The five-architecture CPU
smoke completed successfully.

The one-epoch smoke is a pipeline check only. Its fresh gaps are negative or
zero and do not establish LoP absence, model utility or architecture ranking.
scientific_conclusion_allowed=false.
