# Remote validation receipt

Recorded on 2026-11-04 against unified base
`24083a5e681af291a092a07e01db60962ca964cf` and the peer revisions in
[summary.json](summary.json). The implementation source hashes in that file
identify the code used for these runs. The historical key
`uncommitted_source_sha256` describes its state when the runs were performed.
The [source comparison](source-comparison.json) records the implementation and
focused test hashes at qualification and PR preparation. Three files received
trailing-blank-line cleanup; every other byte matched. Documentation and saved
receipts were added during PR preparation.

## Checks completed

- [Focused tests](tests.txt): 36 passed, with the source archive available.
  This includes `tests/test_henaff2021.py` and `tests/test_task_protocol.py`.
  The suite was rerun successfully against the final PR source after cleanup.
- [Full-run qualification](qualification.json): 400 distinct images produced a
  neural assembly of 400 presentations by 38,400 features. Two complete CLIP
  runs produced exactly matching neural values and stimulus IDs.
- [Analysis summary](summary.json): all 40 trajectories were valid at strides
  1 and 2. It contains source and checkpoint hashes, environment versions,
  separate natural/artificial comparisons, and pixel-baseline diagnostics.
- The biological reference checks recover the published retained counts and
  medians and agree with the paired Figure7e source-workbook values.

The tests were run remotely with:

```bash
HENAFF2021_ARCHIVE=/path/to/archive python -m pytest \
  tests/test_henaff2021.py tests/test_task_protocol.py -q
```

All preparation, inference and tests ran on a remote worker. These receipts
describe the focused source-data and model qualification; they do not claim
that every repository test ran. Source-data tests skip in CI when the archive
is absent. Operational host paths are omitted from the shared qualification
receipt. Images, raw biological data, weights and neural recordings are not
included in this directory.

## Resumption checklist

This work is parked. Before proposing a scored benchmark:

- Resolve the discrepancy between the archived pixel baselines.
- Specify the representation metric and how it relates to the biological
  noise-corrected estimates; obtain the authors' inference code if reproducing
  those estimates from raw spike counts.
- Specify spatial calibration and the intended presentation/timing scope.
- Define reliability, ceilings, null-model controls and scoring semantics.
- Requalify against the target UMI and pinned peer revisions, and require
  passing CI before merging.

The current output is descriptive agreement, not a normalized Brain-Score
score. At stride 2 the natural-sequence model median is positive despite its
high correlation with macaque movie-level estimates; see the signed values
in the summary and the [experiment guide](../README.md).
