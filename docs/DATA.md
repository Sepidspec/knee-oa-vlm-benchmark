# Data and fixed allocations

Download the radiographs from the original
[Kaggle dataset by Shashwat Tiwari](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity).
No radiographs are included in this repository. Keep the original folders and
filenames. `data_root` must point directly to the directory containing
`train/`, `val/` and `test/`; it can be left blank for Kaggle input discovery.
Within each split, folders `0` to `4` supply the reference KL grades.

| Split | KL0 | KL1 | KL2 | KL3 | KL4 | Total | Use |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 2,286 | 1,046 | 1,516 | 757 | 173 | 5,778 | Fitting, internal selection, demonstrations |
| val | 328 | 153 | 212 | 106 | 27 | 826 | CAL/DEV allocations |
| test | 639 | 296 | 447 | 223 | 51 | 1,656 | Held-out evaluation |
| auto_test | 604 | 275 | 403 | 200 | 44 | 1,526 | Excluded from the core study |

The frozen allocations use provisional seven-digit filename prefixes:

- Training: 2,889 groups, with the 20% subset containing 578 groups / 1,156 images.
- Calibration: 206 validation groups / 412 images.
- Protocol development: 207 validation groups / 414 images.
- Training selection: internal 80/20 group partitions; the separate CAL/DEV
  labels do not select training hyperparameters.
- S1–S3: three prespecified training-only sets of five images, one per grade;
  S1 remains the main set regardless of test performance.
- Pilot: 50 DEV images, ten per grade.
- Robustness: one index knee per group from 100 test groups.
- Timing: a separately fixed 100-group test sample, three repeated attempts.

The manifest and allocation CSVs are inside the immutable analysis ZIP.
Matching folder counts alone is insufficient to establish the same dataset;
the recorded manifest stores byte/pixel hashes and the run contract links the
audit and configuration. `results/protocol/run_contract.json` records both
the configuration contract and audit fingerprint.

The audit found no cross-split exact byte or pixel duplicates and retained
150,002 cross-split dHash screening flags. Perceptual similarity is a screening
signal, not an automatic duplicate decision. No automatic label corrections
or image exclusions followed these flags. Clinical participant IDs, visit
linkage, reader conventions and foundation-model pretraining exposure remain
unresolved. Do not describe this split as clinically verified patient-independent.
