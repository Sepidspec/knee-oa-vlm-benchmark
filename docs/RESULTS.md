# Reading the recorded results

## RQ1: direct grading

Full-data ResNet-18 achieved QWK 0.7422 and accuracy 62.08%. Frozen P0 achieved
QWK 0 and accuracy 26.99%, selecting KL2 for all 1,656 test cases. The primary
paired difference (P0 minus full CNN) was −0.7422, 95% provisional-prefix
bootstrap interval [−0.7694, −0.7125]. The full CNN still recognized only
18/296 KL1 cases (6.08% recall). Average accuracy hides this class-level weakness.

## RQ2: rubric and demonstrations

Adding the rubric increased QWK by 0.0291 relative to P0
(Holm-adjusted p ≈ 0.0030), but performance remained low and predictions
concentrated on KL0. S1 increased QWK by a further 0.0146 relative to the rubric,
with an interval crossing zero and adjusted p ≈ 0.274. S2/S3 are sensitivity
sets; they do not justify selecting a demonstration set using test scores.

## RQ3: limited-label adaptation

LoRA reached QWK 0.1382, above frozen P0 but below the subset-matched CNN
(QWK 0.5535). LoRA and the 20% CNN had similar exact accuracy (~40.6–40.7%),
while QWK and macro F1 showed much larger differences. No VLM configuration
recognized a KL1 test case. The comparison controls the downstream training
subset; foundation pretraining histories are not matched.

## RQ4: calibration

Temperature scaling improved NLL in all eight conditions in this test snapshot,
with within-family Holm-adjusted tests below 0.05. It did not change the chosen
grade or accuracy. P0's fitted temperature reached the search boundary
(T ≈ 54.60), producing almost uniform five-candidate probabilities. Lower
confidence/ECE does not demonstrate better discrimination or clinical utility.

## RQ5: confidence-based retention

After Holm adjustment across 24 comparisons, full CNN confidence ranking
improved error risk versus the seeded random reference at 60%, 80% and 90%
coverage. The 20% CNN passed at 60% only. LoRA passed at 60% and 80%; it did
not pass at 90%. P0 did not pass at any of these coverages. Do not infer a
significant result merely from a nominal interval or unadjusted p-value.

At the fixed development threshold, the full CNN retained 1,363/1,656 cases
(82.31%), including 459 grading errors; 293 entered review. These are
retrospective benchmark counts, not a validated patient-care release policy.

## RQ6: perturbations and controls

On the fixed 100-group subset, CNN accuracy was 58% for original images and
40% after JPEG quality 60. These changes are descriptive: **none of the 36
perturbation/outcome tests passed Holm adjustment**. Nominal paired intervals
must not be presented as family-adjusted evidence of an effect.

P0 accuracy stayed at 25% and QWK at zero across every image perturbation.
Across the canonical prompt and four paraphrases, 24 cases were stable correct,
71 stable wrong and five changed grade. Original, blank and shuffled controls
all had 25% accuracy. Stable wrong predictions and output collapse can create
apparent robustness; these aggregate controls do not prove that pixels had no
effect on internal scores.

## RQ7: measured resources

Mean warm inference was approximately 4.7 ms for either CNN, 278 ms for P0,
302 ms for P1, 1.03 s for five-example prompts and 296 ms for LoRA. Every one
of the 2,400 measured attempts completed. Model loading, disk access,
instrumentation, quantization and support-image processing boundaries are
documented; these measurements are specific to the recorded runtime.

## Limits on conclusions

The study supports conclusions about this fixed engineering benchmark and
configuration. It does not establish clinical patient independence, external
generalization, robustness to validated acquisition shifts, absence of
foundation-model contamination, or between-seed training stability. The dHash
flags are unresolved screening flags. The validation set had previously been
used by a CNN pilot, as recorded in the frozen protocol. Preserve these facts
in the thesis's limitations.
