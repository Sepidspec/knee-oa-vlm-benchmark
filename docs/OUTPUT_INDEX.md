# Table and figure index

The numbering follows the completed notebook; Tables 1–4 are not part of the
final numbered analysis export. Every figure has a PNG and vector PDF in
`results/figures/`; captions are in `results/figures/captions.json`.

| Research question | Tables | Figures | Supplementary contrast/accounting files |
|---|---|---|---|
| RQ1 — direct VLM versus full CNN | 5–6 | 1–2 | `primary_estimand.json` |
| RQ2 — prompts and training-only examples | 7–8 | 3–4 | `RQ2_paired_contrasts.csv` |
| RQ3 — downstream training budget and LoRA | 9–10 | 5–6 | `RQ3_paired_contrasts.csv`, `RQ3_correctness_transitions.csv`, `training_resources_all_attempts.csv` |
| RQ4 — raw/scaled calibration | 11–12 | 7–8 | `RQ4_calibration_contrasts.csv`, `reliability_15_bins_all_conditions.csv` |
| RQ5 — risk/coverage and fixed thresholds | 13–14 | 9–10 | `RQ5_confidence_vs_random.csv`, `RQ5_release_review_accounting.csv`, `risk_coverage_*.csv` |
| RQ6 — perturbations and prompt/image controls | 15–16 | 11–12 | `RQ6_perturbation_effects.csv`, `RQ6_case_stability.csv`, `RQ6_stability_by_grade.csv`, `RQ6_visual_input_controls.csv` |
| RQ7 — loading, warm latency, memory and components | 17–18 | 13–14 | `logs/timing_*.jsonl` inside the analysis ZIP |

The immutable ZIP also contains the audit manifest, all fixed allocation and
support manifests, predictions for calibration/development/test/controls,
scored test CSVs, training histories, model metadata and environment records.
Its `OUTPUT_CHECKSUMS.json` lists 194 files; the checksum file itself is
excluded from that internal list to avoid self-reference. The outer ZIP hash
is recorded separately in `docs/SOURCE_PROVENANCE.json`.
