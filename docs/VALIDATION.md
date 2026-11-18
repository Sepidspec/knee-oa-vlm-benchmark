# Repository preparation checks

Completed on 18 November 2026:

- The original analysis ZIP passed its SHA-256 and ZIP CRC checks.
- All 194 internal output hashes matched.
- The analysis ZIP contained 41 of 44 protected files; the three omitted
  trained weight files were expected.
- The separate private full backup passed all 44 protected-file hashes.
- The eight original test JSONL files contained the same 1,656 unique eligible
  cases, with reference grades/groups matching the manifest.
- Independent standard-library calculations reproduced accuracy, QWK, macro
  F1, balanced accuracy, MAE and large-error rate in the recorded tables.
- All original and temperature-scaled test probabilities matched the scores.
- All eight timing logs contained 300 successful attempts across 100 unique
  cases and matched the reported warm means.
- All 38 notebook cells passed Python syntax parsing. Notebook metadata was
  checked for valid structure.
- All seven research question analyses ran on a CPU with 2,000 bootstrap
  resamples. Tables 5–18 and the six primary/secondary contrast files matched
  the recorded CSV/JSON outputs within `rtol=1e-8`, `atol=1e-10`.
- A deliberately altered Table 5 was rejected by the checksum verifier.
- No trained weights or source radiographs are in the analysis ZIP. A scan
  for common credential formats found no matches; this is not a guarantee
  about every possible secret format.

See [VERIFICATION.json](VERIFICATION.json) for the point-metric check and
[REPRODUCTION_REPORT.json](REPRODUCTION_REPORT.json) for the CPU reproduction
report and tested package versions. CPU reproduction also passed with versions
different from the recorded study: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and
Matplotlib 3.10.8. The requirements file retains the original study versions.

The full GPU training/inference pipeline was not rerun during preparation.
The publication notebook was prepared from the available source, with changes
listed in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Numeric reproduction
from the immutable prediction snapshot was executed and checked. Figure file
bytes were not compared because plotting environments can alter rendering and
PDF metadata.
