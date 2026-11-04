# Residual polish verification

Changes are uncommitted. The session can write only inside `unified`; the workspace
policy rejected the sibling `core` edits. The core changes below are prepared in
[core-residual-polish.patch](../core-residual-polish.patch), tested in a temporary
copy, and **not applied to the actual core working tree**. Vision and language
remain unchanged. No model weights, benchmark scoring runs, GPU paths, or notebook
executions were used for verification; the requested tests use synthetic fixtures.
The protected `brainscore/validation/` and `docs/audits/` paths were not edited.

## D6: terminology

Inspection found an actual identity alias, `UnifiedModel = Subject`, and
`BrainScoreModel(Subject)`. The public docs already partly explained this, but the
concrete class's one-line docstring could be read as saying it was formerly named
UnifiedModel. The ABC also promised alias retention for only one release. There
is no separate UnifiedModel implementation to delete: core's memory annotations
and vision's `models/clip_vit_b_32_unified/test.py` still use the alias. Both domain
adapters subclass Subject. No public class or import was renamed or removed.

| File | Change |
| --- | --- |
| `docs/concepts.md` | Make the abstract contract, concrete implementation, and identity alias explicit; cite current uses. |
| `docs/umi_api_reference.md` | Lead with class choice and the stable public import path. |
| `EXTENDING.md` | Explain which class to instantiate or annotate before the registry examples. |
| `core/brainscore_core/contract.py` (patch only) | Align the ABC docstring; retain the live alias without an unsupported one-release promise. |
| `core/brainscore_core/brainscore_model.py` (patch only) | Explain the concrete subclass and remove the ambiguous former-name wording. |
| `core/brainscore_core/model_interface.py` (patch only) | Document the three names at the public import facade. |

Observed verification: a Python import check confirmed `UnifiedModel is Subject`
and that BrainScoreModel, VisionModelAdapter, and LanguageModelAdapter all subclass
Subject. Existing compatibility tests against the temporary core implementation:

```bash
PYTHONPATH="$PWD/.polish-core-check" RESULTCACHING_DISABLE=1 /opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest ../core/tests/test_subject_rename.py ../core/tests/test_model_interface_public_api.py -q -p no:cacheprovider
```

Result: **8 passed in 1.18s**. Residual: the core docstrings remain unchanged until
the patch can be applied in a writable core workspace.

## D10: heavy downloads

Inspection confirmed that the heavy factories entered Hugging Face or direct URL
loaders without a disk/intent check. The current activation wrappers already select
CUDA, then MPS, then CPU; the explicit float32 registrations are not evidence of
CPU fallback. Placement and numerical precision were left unchanged.

| File | Change |
| --- | --- |
| `brainscore/models/_downloads.py` | Add a prompt-free guard with local shard checks, free disk, source, approximate checkpoint size, destination, and an environment opt-out. |
| `brainscore/models/blip2_opt_2_7b/model.py` | Guard BLIP-2 before its first loader and propagate cache/local-only options. |
| `brainscore/models/qwen25_vl_3b/model.py` | Guard ordinary and VWFA Qwen-VL registrations and propagate cache/local-only options. |
| `brainscore/models/qwen36_27b/model.py` | Guard the 27B checkpoint and tokenizer without changing automatic placement. |
| `brainscore/models/vjepa/model.py` | Guard V-JEPA v2 weights and processor. |
| `brainscore/models/vjepa_v1/model.py` | Guard the direct checkpoint download before URL retrieval; reuse existing nonempty files. |
| `brainscore/models/multimodal_av_blip2_wav2vec2/model.py` | Guard the heavy BLIP-2 tower before processor or weight loading. |
| `brainscore/models/multimodal_av_qwen_wav2vec2/model.py` | Guard the heavy Qwen-VL tower before processor or weight loading. |
| `brainscore/models/multimodal_av_vjepa_wav2vec2/model.py` | Pass the actual multimodal/null model identifier into the shared V-JEPA guard. |
| `tests/test_model_downloads.py` | Cover cold, partial, complete, malformed, and broken-link caches; low/unreadable disk; opt-out; both cache API shapes; registered-factory ordering. |
| `tests/test_registered_dyslexia.py` | Opt out only in the existing CPU-toy fixture, whose loaders are all mocked; preserve every assertion. |
| `docs/model_downloads.md` | Document opt-out, cache paths, scope, estimate sources, and limitations. |
| `docs/umi_api_reference.md`, `EXTENDING.md` | Link the guard at the load/registration entry points. |

Set `BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1` for managed CI/EC2 downloads. It
bypasses both the opt-in and the conservative disk budget; it never overrides
Hugging Face offline mode. Otherwise complete cached weights load locally, and a
cold or partial cache raises actionable instructions without waiting for input.

Observed targeted command:

```bash
RESULTCACHING_DISABLE=1 /opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests/test_model_downloads.py -q -p no:cacheprovider
```

The initial 28 cases passed in 8.07s; the later cache-API fallback case is covered
by the final full-tier run below. The first full run exposed two CPU-toy fixtures
blocked by the new real-cache guard; the fixture-specific opt-out above fixes
that mismatch without bypassing the guard for other tests.

Residual: sizes are estimates of a full checkpoint, not remaining bytes in a
partial cache or guarantees of RAM/MPS fit. The documented scope excludes smaller
models, auxiliary audio towers, and legacy domain factories. Hugging Face sizes
were inspected from publisher file listings linked in `model_downloads.md`; the
V-JEPA v1 full-training-checkpoint size is approximate. A metadata-only HEAD
request for v1 failed because this shell could not resolve the host; no bytes were
downloaded. Actual heavy loading, device execution, and performance are untested.

## D13: peak memory

Inspection found that `get_peak_memory()` still returned instantaneous RSS on
CPU/MPS while claiming a resettable peak. However, `check_memory()` no longer
consumes that helper: it computes host array/workspace estimates from shapes. Its
RSS reading is a baseline, not a measured peak. The actual remaining MPS risk is
that forward-pass and driver allocations share RAM and are outside that model.

| File | Change |
| --- | --- |
| `core/brainscore_core/memory.py` (patch only) | Use the existing OS RSS high-water mark for CPU/MPS peak reporting, document lifetime/reset scope, and warn at every MPS check that host estimates do not establish runtime fit. |
| `core/tests/test_memory.py` (patch only) | Add seven cases for macOS/Linux high-water units, non-resettable host peaks, optional torch, unchanged CUDA semantics, and MPS/CPU warning behaviour. |

The patch makes the **process RSS peak real**, not the total MPS device/driver
peak. The OS exposes a lifetime high-water mark, not a resettable per-run mark;
that is why the helper documents its scope and `check_memory` warns separately.
The estimator's formulas and rejection threshold are unchanged.

Observed temporary-copy command:

```bash
PYTHONPATH="$PWD/.polish-core-check" RESULTCACHING_DISABLE=1 /opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest .polish-core-check/tests/test_memory.py .polish-core-check/tests/test_model_interface.py -q -p no:cacheprovider
```

Result: **157 passed, 9 warnings in 1.88s**. A separate import-path assertion
confirmed that these tests loaded the temporary core package. Accelerator APIs
in the new tests are mocked; no GPU workloads were executed.

```bash
git -C ../core apply --check '/Users/kartik/Brain-Score Unified/unified/core-residual-polish.patch'
```

Result: exit 0; the patch applies cleanly. Residual: **D13 is not installed in the
actual core working tree**. Even after applying it, actual MPS device/driver peak
measurement remains outside this polish change; an estimate that passes is not
an assurance against OOM.

## C3: two notebooks

`08_brain_visualization.ipynb` is already self-contained in the current checkout.
Its only executable imports are NumPy and matplotlib; it reads the committed
1000-parcel network array and displays the committed cortical PNG in Markdown.
There is no renderer/fetch call in its code cells. **No change was made.** The
standalone `cortical_surface_map` function still fetches anatomy if separately
called; notebook 08 does not call it.

| File | Change |
| --- | --- |
| `notebooks/12_embodied_vlm_game.ipynb` | Mark the 11/20 versus 12/20 success-rate comparison as a negative result, explain success-conditional efficiency and training accuracy, and distinguish plausible looping from an observed diagnosis. |

Notebook 12 already discussed wedging, but treated efficiency as decisive without
disclosing that `evaluate_policy` averages it only over solved games. It also
claimed the escape heuristic removes wedging despite two remaining failures.
The new narrative preserves the recorded numbers and does not infer significance
or a causal diagnosis from 20 aggregate outcomes.

Observed verification used an inline command with the specified interpreter to
run `nbformat.validate`, compare every code cell with `git show HEAD:<notebook>`,
parse notebook 08's imports with `ast`, load the local `.npy` with NumPy, and verify
the PNG with Pillow. Result: both schemas valid; notebook 08 unchanged; its array
has 1000 entries spanning all seven networks; its PNG is valid; all notebook 12
code, execution counts, and outputs are identical to HEAD. The final schema and
preservation check was:

```bash
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
import nbformat
for name in ('08_brain_visualization', '12_embodied_vlm_game'):
    filename = f'notebooks/{name}.ipynb'
    nbformat.validate(nbformat.read(filename, as_version=4))
    current = json.loads(Path(filename).read_text())
    original = json.loads(subprocess.check_output(['git', 'show', 'HEAD:' + filename]))
    assert [c for c in current['cells'] if c['cell_type'] == 'code'] == [c for c in original['cells'] if c['cell_type'] == 'code']
    if name.startswith('08'):
        assert current == original
    print(name + ': valid; code, counts and outputs preserved')
PY
```

Both notebooks reported valid with code, counts, and outputs preserved. Neither
notebook was executed. Residual: the empirical policy results were not reproduced, and more
seeds/failure traces would be needed for statistical or causal claims.

## Final fast-tier verification

```bash
cd "/Users/kartik/Brain-Score Unified/unified" && RESULTCACHING_DISABLE=1 /opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests -m "unit or integration" -q -p no:cacheprovider
```

Observed result on the actual working tree and installed core:

```text
718 passed, 64 deselected, 72 warnings in 67.92s (0:01:07)
```

The original 689 cases remain green, with 29 new offline download-guard cases.
`git diff --check` passed, changed Python sources parsed successfully, the core
patch passed `git apply --check`, and the protected paths are absent from the diff.

The unified fast tier also passed with the prepared core implementation on the
Python import path:

```bash
PYTHONPATH="$PWD/.polish-core-check" RESULTCACHING_DISABLE=1 /opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests -m "unit or integration" -q -p no:cacheprovider
```

```text
718 passed, 64 deselected, 72 warnings in 65.96s (0:01:05)
```

This verifies the combined proposed changes without modifying the actual core
repository. Temporary copies were removed after verification; the commands above
record their paths at test time. The patch and this report remain for review.
