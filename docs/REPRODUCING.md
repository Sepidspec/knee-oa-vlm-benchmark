# Reproducing the study

## 1. Integrity check

From the repository root:

```bash
python scripts/verify_results.py
```

Python 3.10 or newer is recommended. No third-party package is needed for this
step. The immutable analysis ZIP is verified against its recorded SHA-256,
then all 194 internal output checksums are verified. Of the 44 frozen files,
41 are included and three trained weight files are intentionally excluded.
The verifier also independently recomputes point metrics and validates all
8 × 1,656 original test predictions and 8 × 300 timed attempts.

The provenance of the source notebook, publication edits and original result
archive is recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).
The recorded test contract remains unchanged. A fresh run has a new contract
because its configuration explicitly pins the model revision; this does not
rewrite the historical snapshot.

## 2. Reproduce analysis from saved predictions on a CPU

Install `requirements-analysis.txt` in a virtual environment, then run:

```bash
python scripts/reproduce_analysis.py --output reproduced
```

This extracts the verified snapshot into the selected output directory and
runs the same numerical function bodies used in notebook sections 6–10 and
12–13. It does not execute model training, image inference or timing again.
The default is the recorded 2,000 bootstrap resamples. RQ5 can take several
minutes because cases are ranked again within each bootstrap resample.

The script compares Tables 5–18 and all primary/secondary contrast files
against the recorded snapshot (`rtol=1e-8`, `atol=1e-10`). The report is saved
to `REPRODUCTION_REPORT.json` in the output directory. The output directory
must be new or empty. Original data and archive bytes are preserved.

For one research question:

```bash
python scripts/reproduce_analysis.py --rq 5 --output reproduced_rq5
```

For a quick smoke check with fewer resamples:

```bash
python scripts/reproduce_analysis.py --bootstrap 20 --output reproduced_smoke
```

Intervals from the smoke check are not the recorded study intervals. No
numeric equivalence check is claimed for a changed bootstrap count. Figure
PDF/PNG bytes may change with fonts or plotting versions; the comparisons
concern numerical tables and contrasts.

## 3. Reproduce training, calibration, test inference and controls on Kaggle

1. Import `notebooks/Knee_OA_Study.ipynb` into Kaggle.
2. Attach the dataset described in [DATA.md](DATA.md).
3. Select a GPU accelerator. The recorded hardware was one Tesla T4. Enable
   Internet if dependencies and the base-model snapshot are not already cached.
4. Run section 1's environment check. If it reports a required restart,
   restart the Python kernel and rerun the environment cell before continuing.
5. In section 2, keep `mode='study'`, set a **new** `output_root`, and leave
   `resume_from=''` for a fresh run. The default is
   `/kaggle/working/knee_oa_reproduction`.
6. Run sections 2–5 definitions. Section 3 performs the image audit and writes
   the fixed allocations. Section 5a is a pilot and is skipped in study mode.
7. Run section 5b: CNN selection/refit, calibration/development predictions,
   and limited-label LoRA fitting. Selection uses internal training partitions.
8. Run section 5c: fit temperatures on CAL, choose review thresholds on DEV
   and freeze protected inputs **before** held-out test inference.
9. Run section 5d for all eight fixed test configurations, then sections 6–13
   for RQ1–RQ7, controls and measurements.
10. Run section 14 to export the analysis ZIP. Preserve the full working
    study directory privately if future model inference/resume is needed.

The notebook pins Qwen revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. The historical configuration used
`model_revision='main'`, but `protocol/model_revision.json` records the exact
resolved revision and the model loader reuses it on resume. ResNet-18 uses the
torchvision ImageNet initialization. Model downloads and data access remain
subject to their providers' terms.

The notebook setup controls the Transformers/PEFT/bitsandbytes combination.
Do not replace a working matched CUDA torch/torchvision pair blindly with CPU
wheels. Recorded packages and GPU details are in
`results/protocol/environment.json`; the complete observed environment is
inside the analysis ZIP. The recorded environment is evidence of the run,
not a universal lockfile guaranteed to install on every operating system.

## 4. Resume an existing frozen study without retraining

Use the **full** backup containing the three trained weight files. The public
analysis ZIP omits those files and is sufficient only for cached analysis.

1. Restore the full ZIP to a writable directory on Kaggle, or attach a private
   prior study directory as a Kaggle dataset. The directory passed below must
   directly contain `protocol/`, `checkpoints/` and `predictions/`.
2. Set `resume_from` in section 2 to that study directory. Choose a separate
   writable `output_root`; do not use the same directory as `resume_from`.
3. Run the setup and sections 2–5 **definitions**. The publication notebook
   loads the saved configuration before calculating the contract, copies only
   missing files, and saves the resumed environment under `logs/`.
4. When it prints `Frozen study restored`, the protected file verification
   has passed. **Skip sections 5a, 5b and 5c.**
5. Continue with 5d only if test predictions are incomplete. The JSONL cache
   skips completed case IDs. If inference is complete, continue directly with
   the relevant analysis cells.

Do not use **Run All** for this resume path: it would also execute the
development/freeze cells. Do not alter frozen prompts, allocations, weights,
temperatures or thresholds using held-out test outcomes. A changed experiment
requires a separate run directory and contract.

You can independently verify a private full backup on a computer with Python:

```bash
python scripts/verify_results.py --archive /path/to/full_backup.zip --require-weights
```

## 5. Create another private full backup

The backup utility verifies all frozen files before including the entire
study directory, including CNN and LoRA weights:

```bash
python scripts/backup_study.py /kaggle/working/knee_oa/study \
  --output /kaggle/working/Knee_OA_full_study_backup_new.zip
```

Keep this backup private. It is deliberately ignored by Git. Only the
original analysis ZIP is allowlisted for this repository. If a notebook ZIP
link fails, use Kaggle's output/file browser to download the file from the
working directory; a failed proxy link does not imply that computation was lost.
