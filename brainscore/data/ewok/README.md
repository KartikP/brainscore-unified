# Prepare EWoK data locally

EWoK supplies pairs of contexts and target statements, with the correct pairing
for each item. This plugin converts the authors' files into a validated local
dataset used by both EWoK benchmarks.

## 1. Install the file readers

After installing the four UMI packages, run from the unified repository root:

```bash
# Install the file readers and the hf command, within the pinned versions
python -m pip install -c install/v2-constraints.txt -e '.[ewok-data]'
```

## 2. Choose a source

Put the native files in the source folder (`~/brainscore-data/sources/ewok`, or
give another with `--source`) and package them:

```bash
# Build from the files in ~/brainscore-data/sources/ewok into the folder the benchmarks read
python -m brainscore.data package ewok
```

The source is a native CSV or Parquet file, or a directory containing those
tables. Keep the authors' text and columns unchanged.

Required columns are `Domain`, `MetaTemplateID`, `TemplateID`, `Version`,
`Context1`, `Context2`, `Target1`, and `Target2`. Column casing and underscores
may differ. A missing `Version` can come from a directory name such as `vers=0`.
Use one native export; do not mix it with model evaluation outputs.

A [paper checkout](https://github.com/ewok-core/ewok-paper) containing
`analyses/data.zip` and `config.zip` also works as the source. Use revision
`9e40d30e242925866ee50448f80a13bcdf971318` and set `EWOK_ARCHIVE_PASSWORD` to the
password supplied under the authors' terms. The builder reads their corrected
inclusion table and applies their final exclusions.

Hosted files: obtain approval for the
[Hugging Face dataset](https://huggingface.co/datasets/ewok-core/ewok-core-1.0),
authenticate with `hf auth login` or `HF_TOKEN`, and download the pinned revision
into the source folder:

```bash
# Download the pinned native table into the source folder (needs approved access)
hf download ewok-core/ewok-core-1.0 data/test/ewok-core-1.0.parquet \
  --repo-type dataset --revision 34d912a608066c92e2990a0328ffc3bd9a716042 \
  --local-dir ~/brainscore-data/sources/ewok
```

## 3. Where the build goes

The command writes to the folder the benchmarks read: `~/.brainio/ewok-core-1.0`,
or the folder that `BRAINSCORE_EWOK_ROOT` names when it is set. `--output` writes
elsewhere; then set `BRAINSCORE_EWOK_ROOT` to that folder. The output directory
must be new. A successful build contains:

| File | Purpose |
| --- | --- |
| `items.json` | Validated items with stable IDs |
| `manifest.json` | Counts, source hashes, exclusions, and the items checksum |
| `build.json` | The size, sha1, content summary and check result of each file |

Check that the dataset loads without loading a model:

```python
import brainscore

# Load the build; the loader checks its checksum and item count
dataset = brainscore.load_dataset("EWoK-core-1.0")
print(dataset.sizes)
```

The loader checks the saved checksum and item count. The data contains answer
keys, not human recordings. Keep EWoK text and derived recordings private under
the authors' distribution terms. A successful build does not establish paper
replication; the paper archive currently yields 4,372 items, versus 4,374 stated
in the paper.

## 4. Run the benchmarks with your model

Both benchmarks run on the common channels; each question is one stimulus row.

`EWoK-core-1.0-logprobs` shows each context as `text`, exactly as given, with
both targets as the options of that row. A model with a `likelihood` part
returns the log probability of each target on `behavior:log_probabilities`: the
summed log probability of the target's tokens in `f"{context} {target}"`, plain
text, no template, as EWoK's `score` does. Only a `likelihood` part gives that
format, so a model without one does not declare it and is N/A.
`chat_parts` and `build_causal_lm_subject` give such a part; so does
`EWoKProvider(native_model)` from `brainscore.harnesses.ewok`, which calls the
authors' own `score`.

Log-probability scores can differ slightly from EWoK's own evaluator, for two
reasons. EWoK sets the logit of the padding or end-of-sequence token to minus
infinity before it normalizes; `chat_parts` and `build_causal_lm_subject` do
not. EWoK tokenizes with `add_special_tokens=True`, so some tokenizers add a
start token before the context; these parts add none. The score says so in
`score.attrs['input_adaptation']`.

`EWoK-core-1.0-choice` shows EWoK's own choice prompt as `text`, with no
sentence added, and asks for `behavior:choice` between 1 and 2. A model with a
`reply`, `likelihood` or `answer` part runs it.

```python
# Load the benchmark, then score your model on it
benchmark = brainscore.load_benchmark("EWoK-core-1.0-logprobs")
score = benchmark(model)
```

Choice scores can differ from EWoK's own evaluator. A chat model answers
through its own chat template and the model decodes by its own rule, where EWoK
sends the raw prompt and limits greedy decoding to 1 or 2. The score says so in
`score.attrs['input_adaptation']`.

## Rebuild

### ewok

- Source: EWoK-core-1.0 (Ivanova et al. 2024, https://arxiv.org/abs/2405.09605): https://huggingface.co/datasets/ewok-core/ewok-core-1.0 at revision 34d912a608066c92e2990a0328ffc3bd9a716042, or the paper checkout https://github.com/ewok-core/ewok-paper at revision 9e40d30e242925866ee50448f80a13bcdf971318. Licence: the authors' terms of use (https://github.com/ewok-core/ewok-paper/blob/main/TERMS_OF_USE.txt); do not publish the text.
- Access: gated: request access on the Hugging Face page, then sign in with `hf auth login`. The paper checkout needs the password of the terms of use in EWOK_ARCHIVE_PASSWORD.
- Needs: `pip install -e '.[ewok-data]'` (file readers and the `hf` command); under 32 MB; minutes.
- Command: `python -m brainscore.data package ewok` with the source in `~/brainscore-data/sources/ewok` (or `--source <folder>`): the native table as `hf download ewok-core/ewok-core-1.0 data/test/ewok-core-1.0.parquet --repo-type dataset --revision 34d912a608066c92e2990a0328ffc3bd9a716042 --local-dir <source folder>` writes it (native CSV or Parquet files also work), or the paper checkout with `analyses/data.zip` and `config.zip`. Output (or `--output <new folder>`): the folder the benchmarks read, `~/.brainio/ewok-core-1.0` (or `$BRAINSCORE_EWOK_ROOT`).
- Writes: `items.json` (one item per contrast), `manifest.json` (counts and the items checksum), `build.json`.
- Check: the printed counts; no sha1 is recorded for these files
