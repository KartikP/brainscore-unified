# Test word meaning through vision

Can a vision encoder represent relationships between words it sees as images?
This experiment tests **country-to-capital retrieval** from
[Vision Transformers Learn Word Meaning from Images Alone](https://arxiv.org/abs/2610.13252).
It does not ask the model to generate answers or measure sentence understanding.

```text
Word image + scrambled-letter control
                 ↓
BrainScoreModel.process(StimulusSet)
                 ↓
VisionWrapper → DINO encoder → final class token
                 ↓
Subtract control activity; average backgrounds; center vocabulary
                 ↓
Rank each capital among every other word by cosine similarity
```

The **class token** is the model's summary vector for an image. No text encoder,
answer classifier, projection head, or fitted mapping is added. The subject
declares only the `vision` input channel. `word_representation` names a recording
target; it makes no claim about a biological brain region.

## Scope

| Setting | This experiment |
| --- | --- |
| Query pairs | The authors' 60 country/capital pairs, country → capital |
| Candidate vocabulary | All 1,561 words, including pseudowords; exclude the query itself |
| Images | The authors' word and phase-scrambled control renderings |
| Backgrounds | 96 per word, or an explicitly labeled pilot subset |
| Features | Final class token, without projection or per-image normalization |
| Arithmetic | Store image features as float16, then subtract and average in float32 |
| Score | Fraction of capitals in the ten nearest words; ties favor the target |
| Models | Hugging Face DINOv2 and DINOv3 ViT encoders, FP32 inference |
| Publication | Raw research score; not eligible for the Brain-Score leaderboard |

This implements one experiment, not the entire paper. Noun/brand categories,
synonyms, word-to-image alignment, finetuning, and rendered-sentence Pereira
evaluation are outside this example. FP32 matches the authors' DINOv3-H setup;
their 7B run uses BF16, so this script is not an exact 7B reproduction.

## Prepare the authors' images locally

Use your own authorized ImageNet access. No ImageNet images, model weights, or
rendered stimuli are included here. An existing author-format data directory is
also accepted; it must contain `lexicon/manifest.parquet` and `renders/`.

```bash
git clone https://github.com/gjoelbye/vision-word-semantics.git
cd vision-word-semantics
git checkout 0651a533d6fe5cb19c007ce9fea9afb9822582ca
```

In a **separate author environment**, follow their
[installation instructions](https://github.com/gjoelbye/vision-word-semantics/tree/0651a533d6fe5cb19c007ce9fea9afb9822582ca#installation)
and [ImageNet download instructions](https://github.com/gjoelbye/vision-word-semantics/blob/0651a533d6fe5cb19c007ce9fea9afb9822582ca/data/README.md).
The UMI environment remains Python 3.11. This experiment does not need SPEECH-COCO.

With the fonts, WordNet, and ImageNet validation data installed:

```bash
export VIT_SEMANTIC_DATA=/absolute/path/to/word-data
python scripts/stimuli/backgrounds.py
python scripts/stimuli/vocabulary.py
python scripts/stimuli/render.py
```

These are substantial preparation steps, not a quick smoke test. Rendering
creates hundreds of thousands of images. Run them once per data directory;
the authors warn that rerunning background selection can change the stimuli.
Keep this directory for both native and UMI comparisons.

## Run through UMI

Return to your UMI Python 3.11 environment and the unified repository root.
Install Parquet support if absent: `python -m pip install 'pyarrow>=14,<26'`.
Obtain authorized access to the DINOv3 checkpoint and choose its immutable
40-character Hugging Face revision. The paper does not pin a model revision.

Start with one background per word. This keeps all retrieval candidates but
is **a pilot**, not the paper's 96-background result:

```bash
python examples/rendered_word_semantics.py \
  --author-repo /absolute/path/to/vision-word-semantics \
  --data-root /absolute/path/to/word-data \
  --checkpoint facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --revision YOUR_40_CHARACTER_CHECKPOINT_SHA \
  --slot-limit 1 \
  --batch-size 8 \
  --output-dir /absolute/path/to/results/dinov3-h-pilot
```

Downloads are disabled by default; add `--allow-download` to fetch the pinned
checkpoint and processor. The vision wrapper selects CUDA, MPS, or CPU according
to availability. DINOv3-H is large; qualify device memory and runtime on the pilot
before removing `--slot-limit`. The script does not provision cloud resources.

Outputs:

- `result.json`: raw score, each query's rank, data hash, checkpoint revision,
  processor class, package versions, device, and completion status.
- `word_directions.npz`: the averaged, centered activity for every word.
  These are not complete per-image or per-layer recordings.
- `inputs.csv`: word/control pairings, local image paths, and file checksums.

The runner refuses to overwrite an existing output directory. It checks input
pairing and file contents before loading weights. Subsets, changed images, or
different query pairs receive different benchmark identifiers. Failed runs are
recorded as failed, not successful scores. For very long runs, extraction caches
may use substantial disk space; configure `RESULTCACHING_HOME` or disable caching
with `RESULTCACHING_DISABLE=1`.

## Use a different vision subject

Construct `RenderedWordRetrieval(manifest=..., pairs=...)` with local word/control
images. A subject must accept vision stimuli and map `word_representation` to the
features you want to compare. The supplied DINO builder uses the same
`BrainScoreModel`, `VisionWrapper`, and `process()` interfaces as other UMI models.
Changing the model or recorded layer makes a new scientific comparison; it does
not establish replication of a paper checkpoint.

## Verification

```bash
UMI_TEST_CPU_ONLY=1 RESULTCACHING_DISABLE=1 \
  python -m pytest tests/test_rendered_word_semantics.py -q
```

Offline tests cover pairing, subtraction, centering, full-vocabulary ranking,
ties, output alignment, changed-input identifiers, and invalid inputs. A tiny
randomly initialized DINOv2 and DINOv3 compare direct features with those returned through
UMI. That check verifies extraction, not learned word meaning.

To also compare the aggregation and ranks with the pinned authors' actual code
on identical synthetic features (no model download):

```bash
UMI_WORD_AUTHOR_REPO=/absolute/path/to/vision-word-semantics \
  UMI_TEST_CPU_ONLY=1 RESULTCACHING_DISABLE=1 \
  python -m pytest tests/test_rendered_word_semantics.py -q
```

Without that checkout, the reference check is skipped. CI runs the offline
checks with `-m 'not slow'`; the opt-in reference check carries the `slow` marker
because it requires an external checkout, even though its computation is small.

Before claiming a scientific replication, run trained DINOv3-H and the authors'
native pipeline on the **same images, checkpoint, processor, and precision**.
Compare per-image features, centered word directions, and all 60 ranks. Then
compare the aggregate with the published result. Matching the method alone does
not establish numerical reproduction.
