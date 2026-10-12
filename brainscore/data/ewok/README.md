# Prepare EWoK data locally

EWoK supplies pairs of contexts and target statements, with the correct pairing
for each item. This plugin converts the authors' files into a validated local
dataset used by both EWoK benchmarks.

## 1. Install the file readers

After installing the four UMI packages, run from the **unified repository root**:

```bash
python -m pip install -c install/v2-constraints.txt -e '.[ewok-data]'
```

## 2. Choose a source

**Files you already have:** provide a native CSV or Parquet file, or a directory
containing those tables. Keep the authors' text and columns unchanged.

```bash
python -m brainscore.data prepare EWoK-core-1.0 \
  --source /path/to/native-data \
  --output /path/to/ewok-build
```

Required columns are `Domain`, `MetaTemplateID`, `TemplateID`, `Version`,
`Context1`, `Context2`, `Target1`, and `Target2`. Column casing and underscores
may differ. A missing `Version` can come from a directory name such as `vers=0`.
Use one native export; do not mix it with model evaluation outputs.

A [paper checkout](https://github.com/ewok-core/ewok-paper) containing
`analyses/data.zip` and `config.zip` also works as `--source`. Use revision
`9e40d30e242925866ee50448f80a13bcdf971318` and set `EWOK_ARCHIVE_PASSWORD` to the
password supplied under the authors' terms. The builder reads their corrected
inclusion table and applies their final exclusions.

**Hosted files:** obtain approval for the
[Hugging Face dataset](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)
and authenticate with `hf auth login` or `HF_TOKEN`. Then run:

```bash
python -m brainscore.data prepare EWoK-core-1.0 \
  --request-id ewok-core/ewok-core-1.0 \
  --output /path/to/ewok-build
```

Here `--request-id` is the dataset repository ID, not an approval code. The
builder downloads revision `34d912a608066c92e2990a0328ffc3bd9a716042` using your
existing access. It cannot grant access or accept terms for you.

## 3. Point UMI at the build

```bash
export BRAINSCORE_EWOK_ROOT=/path/to/ewok-build
```

The output directory must be new. A successful build contains:

| File | Purpose |
| --- | --- |
| `items.json` | Validated items with stable IDs |
| `manifest.json` | Counts, source hashes, exclusions, and the items checksum |
| `build.json` | How the files were obtained and converted |

Check that the dataset loads without loading a model:

```python
import brainscore

dataset = brainscore.load_dataset("EWoK-core-1.0")
print(dataset.sizes)
```

The loader checks the saved checksum and item count. The data contains answer
keys, not human recordings. Keep EWoK text and derived recordings private under
the authors' distribution terms. A successful build does not establish paper
replication; the paper archive currently yields 4,372 items, versus 4,374 stated
in the paper.

For model setup, scoring, and tools, see [the EWoK guide](../../../docs/ewok.md).
