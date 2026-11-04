# Real-source pilot validation

This receipt covers one scene, seven motion modulation IDs and four flips:
28 trials, with three participants' reference vectors. It is not a full-study
result. All packaging, tests, inference and validation ran on a remote worker.
The final focused regression suite passed **57 tests** in 32.42 seconds.

The separately named Qwen2.5-VL-3B video example consumed all sixteen frames at
30 Hz. RecordBehavior and RecordNeural captured the same 28 presentation IDs.
The recorded V1-mapped model activity has shape 28 x 8 x 1280, with eight
temporal patches and 1,280 units. The mapping is illustrative, not a validated
neural alignment result. The original study supplies human behavior only.

The first parser accepted bare JSON only. All 28 replies contained valid
numerical objects inside JSON code fences, and were rejected by that policy.
strict-json-v1-summary.json preserves that result. The explicit motion-json-v2
policy accepts either bare JSON or one whole-response JSON fence, records the
format, and changes no numerical values. The new summary and CSVs were computed
from saved raw replies; model inference was not repeated. All 28 reports parse.
This format policy was chosen during development, not preregistered.

The model's mean endpoint error is approximately 9.579 displayed pixels/frame,
close to the no-motion control's 9.580. Only two of seven averaged model vectors
have a defined direction; their six participant comparisons have a mean angular
error of 88.37 degrees. These small-pilot values establish that the comparison
runs. They do not reproduce the paper's model results or establish a general
perception effect.

qualification.json checks finite recorded activity, matching presentation IDs,
eight temporal bins, exact agreement with the saved analysis, and invariance to
reversing prediction/reference row order. It records checkpoint SHA-256s and
source-file hashes. tests.txt contains the final focused regression result.

Data source/build manifests and source-alignment receipts live with the data
plugin in brainscore/data/sun2025/data_packaging/validation. Native source files,
prepared clips and full UMI recordings are retained on the worker. Large data
and model files are not embedded in this source tree.

## Notebook and synchronized replay

Validated on 2026-11-04 using the same remote environment. The replay and
notebook-validator tests passed 17 checks in 5.14 seconds; see
`notebook-tests.txt`. The checks cover stimulus identity, frame-to-bin mapping,
missing/invalid timing, preservation of the original activity and raw answer,
and notebook classification.

`notebook-validation.json` records successful fresh-kernel execution of both
notebook modes, with the source notebook's SHA-256. Run mode completed a new
four-trial Qwen experiment with both recorders; replay mode loaded four trials
from the saved 28-trial pilot. Each produced a 4 x 8 x 1280 activity assembly,
four valid numerical reports and a standalone replay HTML file. These runs
verify the workflow, not a full-study scientific effect.

`browser-validation.json` records checks of the fresh run's HTML in Chromium
141.0.7390.37, including its SHA-256. Frame stepping and seeking selected the
correct recorded bins; play/pause, speed selection, raw-value hover and answer
display worked. The narrow layout had no horizontal overflow. The same HTML
worked inside a sandboxed iframe with the notebook's script permission. There
were no page errors or network requests. Desktop, narrow and embedded
screenshots were also visually inspected.

Reproduce these checks with `examples/sun2025/validate_notebook.py` and
`examples/sun2025/validate_replay_browser.py`. Executed notebooks, exported HTML
and screenshots are retained separately from the clean source notebook.
