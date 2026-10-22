# Targeted integration tests

These profiles test real models and datasets outside the short offline suite.
The [selection file](../install/integration-tests.json) defines eight profiles
with 52 test IDs. All eight have an executable runner. Each run checks the
four installed candidate packages before and after testing.

**CPU qualification is partial.** The October 22 fresh-Conda pilot passed
`language-readout`, `language-installation` and `native-text`. Both vision
profiles failed; see the [pilot results](qualification/2026-10-22-cpu-pilot.md).
Other profiles and CUDA remain unqualified.

## Choose a profile

| Profile | What it checks | Prepared caches |
| --- | --- | --- |
| `vision-public-score` | Pixels model on public MajajHong IT data | BrainIO |
| `vision-layer-mapping` | AlexNet layer selection, scores and saved mappings | Torch; BrainIO; ImageNet for PCA |
| `language-public-score` | DistilGPT-2/Futrell reference score | Hugging Face; BrainIO |
| `language-readout` | Reading-time outputs and token handling | Hugging Face |
| `language-installation` | Installed scoring with `yes`/`no` dependency preferences and the CLI | BrainIO |
| `native-text` | CLIP/GPT-2 extraction and `BrainScoreModel.process()` dispatch | Hugging Face |
| `native-vlm` | Qwen image features, patch grouping and `process()` dispatch on CUDA | Hugging Face |
| `behavioral-roar` | Fixed splits, CLIP readout and score structure | Hugging Face; authorized Yeatman2021 BrainIO build |

Model changes also need that plugin's tests and a declared compatible benchmark.
Benchmark changes need a representative model and a cheap baseline where useful.
Data changes need preparation/loading tests and a consuming benchmark. Shared
code changes need the affected profiles in both domains. Combine selections
when a change crosses these boundaries.

These profiles complement [scientific parity and release qualification](release_test_matrix.md).
They do not qualify every plugin, trained DROID, or audio/video scoring.

## Installation coverage

`language-installation` runs the installed command from outside the source tree.
Its reference plugins have no extra requirements: the profile checks scoring
under `yes` and `no`, not pip dependency resolution. Core separately tests that
plugin installation uses the current Python interpreter and stops on pip errors.

The October 22 pilot manually created a fresh Conda environment and passed
installed language scoring. **Automatic Conda creation by the scoring API remains
unqualified.** The old `newenv` case passed
`conda_active=True`, bypassing environment creation. A replacement must create
and clean up a disposable environment, preserve all four candidate wheels, and
run the reference score. The manifest tracks this separately in `unqualified_paths`.

## Prepare a run

Use a disposable Linux/Python 3.11 environment with the four coordinated wheels
installed. Follow [coordinated CI](coordinated_ci.md) to resolve revisions and
build wheels. Keep clean source checkouts under `core/`, `vision/`, `language/`
and `unified/`; they supply the tests, not the imported packages.

Prepare authorized assets separately. A cache directory must contain the exact
checkpoint, processor/tokenizer files, and dataset build required by its profile.
Do not include tokens or credentials. Hugging Face caches include their `hub/`
directory and pinned `refs/` and `snapshots/`. BrainIO must contain all assembly
and stimulus files; a partial cache is insufficient.

Create an asset manifest for the selected profile. For example, `native-text`:

```json
{
  "profile": "native-text",
  "assets": {
    "huggingface": {
      "path": "/prepared/native-text/huggingface",
      "sha256": "CONTENT_DIGEST",
      "build": "CLIP snapshot <commit>; GPT-2 snapshot <commit>"
    }
  }
}
```

Compute `CONTENT_DIGEST` with the runner's directory fingerprint. It covers file
names and contents, including snapshot-link targets:

```python
from pathlib import Path
from brainscore.validation.integration import fingerprint

digest, size_bytes = fingerprint(Path('/prepared/native-text/huggingface'))
print(digest)
```

The required cache names are listed in `required_caches` for each profile. Give
each its own path, digest and build description. The runner verifies and copies
them; it does not download assets or modify the prepared originals. Cache hashes
pin the bytes actually tested, including model-name references.

## ImageNet input for vision layer mapping

Legacy layer selection fits PCA on ImageNet images to reduce activation size.
The layer-mapping profile preserves that procedure; it does not reuse fitted PCA
or score caches. ImageNet is not a requirement for every UMI model or benchmark.

Alongside the `brainio` and `torch` entries, add a `model_tools` entry to the asset
manifest. Put your authorized ImageNet validation HDF5 in that directory as
`imagenet2012.hdf5`. It must contain the 50,000-image dataset at `val/images`.
Record its origin/build and compute the directory digest with `fingerprint()`.
A differently named local file can be copied to this filename; no upload or
redistribution is required.

```json
"model_tools": {
  "path": "/prepared/vision/model_tools",
  "sha256": "CONTENT_DIGEST",
  "build": "ImageNet validation HDF5; record the source and file SHA-256"
}
```

The runner checks the HDF5 before copying inputs or running any model. It sets
`MT_HOME` to the private copy and discards inherited `MT_HOME` and
`MT_IMAGENET_PATH` overrides. Keep enough disk space for both the prepared file
and its disposable copy. The Brain-Score HDF5 currently occupies about 9.83 GB.
The runner does not obtain ImageNet access for you or download the file.

The four mapping tests share one fresh six-combination sweep. They still check
structure, score ranges, best-layer selection and saved files. This avoids
repeating the same expensive scoring solely for separate assertions.

## Execute

Run from outside the source checkouts:

```bash
python -I -m brainscore.validation.integration \
  --root /candidate/repos \
  --revisions /candidate/qualification/revisions.json \
  --wheels /candidate/wheels \
  --manifest /candidate/repos/unified/install/integration-tests.json \
  --assets /prepared/native-text.json \
  --profile native-text \
  --output /candidate/qualification/native-text
```

The output directory must be new. Deny outbound network traffic at the worker
level. The runner removes inherited credentials and cache overrides, enables
Hugging Face offline loading, and rejects Python socket connections; those
checks do not sandbox subprocesses.

The runner requires the profile's host-memory allowance to be available and
enough disk to copy assets plus 2 GiB working headroom. CPU profiles disable
accelerator selection. The Qwen profile requires CUDA and checks the extracted
vision module's actual device and float16 precision. CLIP/GPT-2 fixtures record
CPU placement and float32 precision.

## Read the result

`report.json` summarizes preparation, revisions, dependencies, device inventory,
commands, elapsed time and sampled process-tree memory. Collection/execution
JSON, installed-package identities, logs and `tests.xml` retain the details.
A pass requires:

- Exactly the reviewed, nonempty test IDs, checked before fixtures run.
- Every selected test's setup, call and teardown passing; no skip exemptions.
- Installed package files matching all four supplied wheels before and after tests.
- Complete JSON and JUnit reports, with no import from a package source checkout.
- Completion within the profile's shared collection/execution deadline and
  sampled host-memory limit. Timeout or excess memory terminates the process group.

Asset copying and hashing happen before the test deadline. Host memory is sampled,
not an OS-enforced allocation limit; worker/container limits should provide the
outer boundary. GPU OOM is a failure, not a successful partial run. Proposed
limits still need a bounded pilot before scheduling.

Result caching is disabled. Repeated-output checks establish consistency, not a
cache hit; cache performance remains a separate qualification task.

## Remaining activation work

- Supply reviewed asset manifests and run the Linux CPU/CUDA pilots.
- Qualify fresh Conda environment creation separately from the installed language scoring/CLI profile.
- Add explicit audio/video profiles and scientific reference outputs.
- Review supported scope under #6 before deploying scheduled Jenkins jobs.

Core's companion plugin-runner fix retains results for every selected file,
rejects empty or entirely skipped runs, and runs the requested filename.
Submission/database qualification remains separate under #22.
