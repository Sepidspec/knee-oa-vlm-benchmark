# Implemented study methods

## Task and model conditions

The task is current radiographic KL grade 0–4 from a knee X-ray. Inference
receives image pixels and a fixed prompt; target reference grades are used
only by the evaluator. The study does not generate or score free-text clinical
reports, and BLEU/ROUGE are not study outcomes.

Images are loaded through the archived manifest, converted to RGB and resized
to 224 × 224 using bicubic interpolation. ResNet-18 is initialized from the
torchvision ImageNet weights and uses an ordinary five-class cross-entropy
head. The full and 20% CNN conditions are selected and refitted separately.
The learning-rate candidates are 0.0001 and 0.0003, with internal training-only
selection, early stopping and a final refit.

Qwen2.5-VL-3B-Instruct is loaded with four-bit NF4 double quantization and
float16 computation. P0 is the minimal grading prompt. P1 adds a fixed KL
rubric. S1–S3 use P1 with five labeled training-only demonstrations. The VLM
scores the candidate answers `0`, `1`, `2`, `3`, `4` using conditional token
log-likelihood; the largest score selects the grade. Recorded tokenization,
processor configuration and model revision are preserved. These probabilities
are normalized scores among the five candidates, not unrestricted generated
text probabilities or calibrated clinical risk estimates.

The 20% LoRA condition fits language attention q/v projections using rank 8,
alpha 16, dropout 0.05 and accumulation of eight examples. Learning-rate
candidates are 0.0001 and 0.0002, with two epochs and a declared six-hour
selection/refit wall cap. It uses the same 578 training groups as the 20% CNN.
No optional full-data LoRA result is inferred. The exact module matches,
gradient gate and fitted settings are archived.

## Development, freeze and evaluation

Calibration and review thresholds use separate fixed validation allocations.
Temperature is selected by minimizing calibration NLL over log(T) in [−4,4].
A development threshold targets approximately 80% coverage, retaining every
confidence tie. Temperature scaling preserves the selected grade. The
development thresholds are applied unchanged to test predictions; realised
test coverage can differ from 80%.

Before test inference, the notebook records the contract, conditions, prompts,
support sets, model revision, temperatures, thresholds, allocations and hashes
of 44 protected files. Resumable JSONL records retain failure status and
condition metadata. All eight conditions completed every eligible test case
in the recorded study.

## Outcomes and uncertainty

- QWK: quadratic weighted kappa for ordinal agreement.
- Accuracy: exact grade agreement. All-case accuracy counts failures in the
  eligible denominator; conditional grade metrics use valid predictions.
- Macro F1: the unweighted mean of the five per-grade F1 scores.
- Balanced accuracy: the mean recall over supported reference grades.
- MAE: mean absolute difference between reference and selected grade.
- Large error: absolute grade error of at least two.
- Calibration: five-candidate NLL, multiclass Brier score and top-label ECE in
  15 fixed equal-width bins. Reliability display bands are identified separately.
- Selective prediction: error and large-error risk among retained cases;
  coverage uses all eligible cases, and failed inference enters review.

Uncertainty uses 2,000 bootstrap resamples of the provisional group prefixes,
keeping both knees together when present. Paired contrasts use common-complete
cases and paired resamples. The tests are approximate two-sided centered
bootstrap tests. Holm adjustments apply within the declared secondary families;
the primary contrast is P0 minus full-CNN QWK. Case-sampling intervals do not
include variation from retraining with other random seeds.

The RQ5 comparison uses one deterministic seeded random ordering as a
descriptive reference; resampling reranks both policies. The perturbation
family consists of six transformations × three measured configurations ×
two outcomes (36 tests). Paraphrase controls separate stable correct from
stable wrong predictions. Blank/shuffled image controls do not by themselves
establish a causal explanation of the network's visual dependence.

## Runtime

Warm batch-one inference is measured on 100 fixed groups with three repeats
and three warmups per condition. CUDA synchronization and component hooks
introduce instrumentation overhead. Target disk reads and loading are separate;
support images are processed on every attempt without prefix caching.
Memory is peak PyTorch-allocated GPU memory, not total device/process memory.
Training GPU-hours are allocated-GPU wall seconds divided by 3,600, not a
measurement of device utilization. The quality cohort and timed cohort are
separately labeled in Table 17.
