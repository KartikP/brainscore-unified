# EWoK: world-knowledge questions with UMI tools

EWoK pairs two contexts with two target statements. Context 1 fits target 1;
context 2 fits target 2. The benchmark measures whether a model distinguishes
those relationships. The data plugin supplies answer keys, not human recordings.

## Prepare your data

For step-by-step setup, see the [data README](../brainscore/data/ewok/README.md).

After the four-repository installation, install the file readers from the workspace root:

```bash
python -m pip install -c unified/install/v2-constraints.txt -e "./unified[ewok-data]"
```

Use files downloaded from the authors, without renaming their columns:

```bash
python -m brainscore.data prepare EWoK-core-1.0 \
  --source /path/to/native-data \
  --output /path/to/ewok-build
export BRAINSCORE_EWOK_ROOT=/path/to/ewok-build
```

Accepted sources:

- A native CSV or Parquet file, or a directory containing those tables.
- A paper checkout containing `analyses/data.zip` and `config.zip`. Set
  `EWOK_ARCHIVE_PASSWORD` to the password supplied in the authors’ terms of use.
  This path uses their corrected inclusion table and final exclusions.

Native tables must identify `Domain`, `MetaTemplateID`, `TemplateID`, and
`Version`, with `Context1`, `Context2`, `Target1`, and `Target2`. Casing and
underscores may differ. A missing version can be read from an author-style
`vers=0` directory. Ambiguous or incomplete files fail before model loading.

For the hosted dataset, first obtain access from the authors and authenticate
with `hf auth login` or `HF_TOKEN`. Then:

```bash
python -m brainscore.data prepare EWoK-core-1.0 \
  --request-id ewok-core/ewok-core-1.0 \
  --output /path/to/ewok-build
```

For EWoK, this identifier is the Hugging Face dataset ID, not an approval code.
The download is pinned to revision `34d912a608066c92e2990a0328ffc3bd9a716042`.
The command does not request access or accept terms for you. Other data plugins
can resolve their own author-issued request IDs through the same command.

The builder preserves native text, validates paired items and unique IDs, and
writes a checksummed manifest. It never replaces an existing output directory.
Files are private to the local user by default. Do not publish EWoK text,
prompts, or derived recordings as plaintext; use the authors’ protected or
gated distribution rules.

## Select an evaluation

| Benchmark | Model operation | Score |
| --- | --- | --- |
| `EWoK-core-1.0-logprobs` | Conditional target log probabilities under each context | Correct comparisons; exact ties earn half credit |
| `EWoK-core-1.0-choice` | Authors’ optimized two-example prompt, constrained 1/2 answer | Correct choices; malformed answers count as incorrect |

Each item contributes two decisions. Scores average within each dataset version,
then average the versions equally. They are raw accuracy, without a human ceiling.
Domain scores, item counts, invalid answers, and the dataset checksum accompany
the result. A domain subset is explicitly recorded.

The choice variant is the authors’ constrained-choice protocol. It is **not** the
Qwen chat-with-thinking localizer pilot shown on the demonstration website.
Changing prompts, chat formatting, or reasoning settings creates a different
experimental protocol and must be reported separately.

## Connect a model

`EWoKProvider` delegates to the authors’ model evaluator, retaining its
tokenization, probability calculation, and choice prompts. In a compatible
EWoK environment, `native_model` is an `ewok.evaluate.model.Model` instance.
Use the paper source revision `9e40d30e242925866ee50448f80a13bcdf971318` and
record model/checkpoint settings in provenance. UMI does not install or run that
source automatically.

```python
import json
import brainscore
from brainscore.harnesses.ewok import EWoKProvider
from brainscore.model_helpers.response_trace import build_trace_subject

subject = build_trace_subject(
    "my-model",
    provider=EWoKProvider(native_model),  # Calls the existing EWoK model evaluator.
    parse=json.loads,  # Reads the returned batch of probabilities or choices.
    provenance={
        "checkpoint": "your exact model revision",
        "evaluator_revision": "9e40d30e242925866ee50448f80a13bcdf971318",
    },
)
benchmark = brainscore.load_benchmark(
    "EWoK-core-1.0-logprobs",
    root="/path/to/ewok-build",
    batch_size=8,
    output_dir="runs/ewok",  # A new directory for this evaluation.
)
score = brainscore.score(subject, benchmark)
```

A custom provider can implement the same request payload instead. Log-probability
requests contain `operation`, `targets`, and `contexts`; choice requests contain
`targets`, `contexts1`, `contexts2`, `gen_type`, and `prompt_type`. Return
`{"text": json.dumps(values)}` with one value per target, in request order.
Do not substitute a model's self-reported confidence for conditional log probabilities.
The request never contains expected answers.

## Attach tools

Use the same `Experiment` pattern as other benchmarks. `benchmark.protocol()`
returns a `CallableProtocol`: the evaluation steps wrapped for use with tools.
`Experiment.run()` executes those steps and returns the score and run directory.

```python
from brainscore.experiments import Experiment, RecordInputsOutputs

result = Experiment(
    subject=subject,
    protocol=benchmark.protocol(),  # Prepares the same evaluation steps and scoring.
    tools=[RecordInputsOutputs()],  # Saves requests, responses, and failures.
    output_dir="runs/ewok-recorded",
).run()
```

Add `TorchInstrumentation` around the actual model and `RecordActivity` or
`ScaleActivity` using its layer paths. The benchmark does not own model hooks.
Native constrained choices do not generate a reasoning trace; `RecordReasoning`
can only save reasoning a provider actually returns.

## Qualification

The registered log-probability benchmark exactly reproduces independent
calculations from the authors’ archived GPT-2 XL, Phi-1.5, and Phi-2 outputs.
This checks the scoring path, not fresh model inference or numerical agreement
of a newly installed model evaluator.

The paper archive yields **4,372 items** after its final exclusions, while the
paper states **4,374**. Both the actual count and source hashes are retained.
Hosted access and its dataset contents require the user's approval and have not
been qualified with a real authenticated download. A local build or matching
rounded score alone does not establish paper replication.

Sources: [paper](https://arxiv.org/abs/2405.09605),
[paper code and terms](https://github.com/ewok-core/ewok-paper),
[hosted data](https://huggingface.co/datasets/ewok-core/ewok-core-1.0).
