# Use UMI tools with CogGym

CogGym supplies the prompts, questions, answer parser, and scoring. Brain-Score connects the model and tools while CogGym runs its own evaluator.

| Component | Role here |
| --- | --- |
| Model | The network or service producing answers. |
| Provider | Your adapter that calls the model using CogGym's arguments. |
| Subject | The UMI interface through which tools observe those calls. |
| Protocol | The procedure that runs the CogGym evaluation. |
| Experiment | Combines the subject, protocol, tools, and output directory. |

## 1. Prepare CogGym

Install the four UMI packages in a separate experiment environment, then pin CogGym:

```bash
git clone https://github.com/lance-ying/coggym.git
git -C coggym checkout a1cd9df1118fec80eba7463de237d1497d77e041
# Keep UMI's pinned numpy, pandas, torch and transformers; stop if CogGym needs others.
python -m pip install -c /path/to/unified/install/v2-constraints.txt \
  -r coggym/evaluation/requirements.txt
python -m pip check
```

If pip reports a conflict, do not drop the constraints: record the conflict and qualify that combination separately.

CogGym imports the Google SDK even for local models; installing it requires no API key. See the [data README](../brainscore/data/coggym/README.md) for the data layout and checks without inference. Loading a benchmark checks the reference revision and prompts, rejects tracked evaluator or data edits, and does not download data.

## 2. Select a benchmark

All 23 public experiments are registered: 11 text, 6 image, and 6 video. Registration does not mean every model has been tested on them.

```python
import brainscore

# Check the experiment before loading a model.
benchmark = brainscore.load_benchmark(
    "CogGym.Hu2023Fine.exp1",
    checkout="/path/to/coggym",
    repetitions=5,  # Match the pinned README example.
    temperature=1.0,
    max_tokens=512,
)
```

Set `BRAINSCORE_COGGYM_CHECKOUT` to omit `checkout`. List available names with:

```python
names = sorted(
    name for name in brainscore.benchmark_registry
    if name.startswith("CogGym.")
)
```

## 3. Connect your model

The following examples assume your integration supplies `provider`. Its CogGym method, `complete_with_metadata(system, messages, model, temperature, max_tokens)`, must preserve the supplied text and media and return a dictionary containing `text`, `reasoning`, and `token_usage`. Use `None` for unavailable reasoning or token counts.

```python
from brainscore.model_helpers.response_trace import build_trace_subject

subject = build_trace_subject(
    "your-provider-model-id",
    # UMI passes one request dictionary; CogGym expects named arguments.
    provider=lambda request: provider.complete_with_metadata(**request),
    parse=str,  # Keep the answer text for CogGym's own parser.
    reset=provider.reset,  # Declare repetition reset once, with the provider.
    input_channels=["generation_request"],  # This example handles text.
    provenance={"checkpoint": "exact revision", "precision": "float32"},
)
```

`provider.reset(repetition)` receives a one-based repetition number. It resets conversation and sampling state without replacing the model or removing its hooks. A stateless provider can explicitly use `reset=lambda repetition: None`. Record the seed policy, checkpoint, precision, decoding, and media-processing settings in provenance.

Declare only request channels your provider implements: `generation_request` for text, `image_generation_request` for text with images, and `video_generation_request` for text with images and video. A multimodal provider can declare all three. Responses use `response_trace`. Benchmarks reject missing media support before inference; a declaration does not implement media processing.

## 4. Run with tools

For a local PyTorch model exposed as `provider.model`:

```python
from brainscore.experiments import (
    Experiment,
    RecordActivity,
    RecordInputsOutputs,
    RecordReasoning,
    TorchInstrumentation,
)

protocol = benchmark.protocol(
    model=subject.identifier,
)
result = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[
        RecordInputsOutputs(),  # Save the model's requests and responses.
        RecordReasoning(),  # Save reasoning only if the provider exposes it.
        RecordActivity(["encoder"]),  # Replace with your model's layer path.
    ],
    instrumentation=TorchInstrumentation(provider.model),  # Connect layer hooks.
    output_dir="runs/coggym-recorded",  # Use a new directory for each run.
).run()
score = result.value
```

For a remote service, omit `RecordActivity` and `TorchInstrumentation`; recording inputs, outputs, and exposed reasoning still works. `benchmark.protocol(...) -> CallableProtocol` means it returns the evaluation procedure. The score is produced when `Experiment.run()` executes it.

The result is CogGym's **raw Pearson R²**, without a human ceiling. Coverage and configuration are in `score.attrs`; artifacts are in `result.directory`. To score without selecting tools, use `brainscore.score(subject, benchmark)` using the reset callback supplied with the subject or benchmark. That path records inputs and outputs automatically.

When scoring without `output_dir`, temporary recordings are deleted on success, failure, or interruption; `score.attrs["run_directory"]` is `None`. Supply `output_dir` when loading the benchmark, or use `Experiment` as above, to retain calls and failed-run evidence. `records_retained` reports that choice on a returned score.

## 5. Compare baseline and intervention

Run separate `Experiment`s for baseline, intervention, and restoration. Keep prompts, model, and generation settings fixed; use a new output directory each time.

Add `Ablate([selection])` to zero selected activity, or `ScaleActivity(targets, factor=0.5)` to halve it. Both use `TorchInstrumentation`. A `Selection` names a layer, optional unit indices, and their axis (`axis=-1` by default). Reuse the same selection for recording and intervention. Zeroing half of one layer is not zeroing half the model.

For a model exposing `model.language_model.layers`, this selects every fifth full block:

```python
from brainscore.experiments import ScaleActivity

targets = [
    f"model.language_model.layers.{index}"
    for index in range(4, len(provider.model.model.language_model.layers), 5)
]
dampen = ScaleActivity(targets, factor=0.5)  # Halve blocks 5, 10, 15, ...
```

Place `dampen` before `RecordActivity` to record the changed outputs. This scales the full block output, including its residual stream; weights stay unchanged. Compare scores on shared scorable questions and report invalid or truncated answers separately. Changed activity does not necessarily mean worse performance.

**Tool scope:** CogGym calls each question a trial. UMI wraps the whole CogGym evaluator as one trial, `external`, in condition `default`. CogGym question IDs are recorded as metadata; they cannot be passed to a tool's `trials=` filter. Likewise, `conditions=["baseline"]` does not create a condition. For the separate runs above, leave tool filters unset. Filtering interventions by individual CogGym questions is not supported.

## Inspect and replay

| Location | Contents |
| --- | --- |
| `experiment.json` | Setup, completion status, and artifact provenance |
| `inputs_outputs/` | Model requests, responses, and recorded activity |
| `reasoning/` | Exposed reasoning or an explicit unavailable marker |
| `coggym/run-001.json` | Reference trial results, summary, scope, and coverage |
| `coggym/analysis.json` | Reference analyzer output, including undefined results |

Call metadata includes experiment, repetition, attempt index, and matching CogGym question IDs. Duplicate prompts can match several IDs; attempt indices are not physical timestamps. Human scoring targets stay in the evaluator and are not sent to the model.

With an explicit output directory, failed, missing, or reordered trials fail the run and preserve their records. Unparseable answers remain visible as unscorable. Undefined correlations also fail registered scoring. Always inspect coverage alongside the score.

### Slider response checks

CogGym's parser can extract numbers from placeholders or refusals. UMI checks slider answers separately, using policy `coggym-slider-v1`. The original responses, parsed answers, and native scores stay unchanged.

A valid slider answer must give one finite number within the declared scale and agree with CogGym's parsed value. For multiple questions, answers must be assigned by question tag, number (such as `"1"` or `"Question 1"`), or an ordered JSON list or sequence of objects. Multi-slider questions must name every option. Placeholders, prose, booleans, nested arrays, duplicate keys, and out-of-range values fail the check. A single bare number or `Answer: 42` is also accepted.

Find `response_validation` in `analysis.json` and under `umi` in each run file. It reports the policy, number of checked and invalid questions, and the trial IDs and reasons for failures. `status: complete` in a run file describes execution; response validity is separate.

The registered benchmark (version 2) returns no score if any slider answer fails. It never repairs or silently drops answers. `CogGymRunner` still returns the native analysis for comparison, with the validation report attached; an invalid report disqualifies that analysis from scientific interpretation. Set an explicit output directory to retain failed-run evidence.

`replay_calls` sends saved requests to a model again. Viewing saved responses needs no inference. Neither reruns an interactive environment. See the [tool guide](experiment_toolbox.md#replay-saved-inputs).

## Advanced: compare direct and UMI execution

`CogGymRunner` delegates to the same evaluator directly, which is useful for checking that tools preserve behavior. Unlike the registered benchmark, it returns an analysis dictionary rather than a `Score`. Both use a 512-token limit. The runner defaults to one repetition (the pinned CLI default); the registered benchmark uses five (the pinned README example). Set comparison settings explicitly:

```python
from brainscore.harnesses.coggym import CogGymRunner

# Match the registered text benchmark above, including its model identifier.
runner = CogGymRunner(
    "/path/to/coggym",
    experiment="Hu2023Fine/exp1",
    model=subject.identifier,
    modalities=["text"],
    repetitions=5,
    temperature=1.0,
    max_tokens=512,
)
direct = runner.run(
    provider,
    reset=provider.reset,
    output_dir="runs/coggym-direct",
)
```

Compare saved requests, responses, parsed answers, and R² against the recorded benchmark run. Controlled decoding and resets are needed for exact equality. Runner `modalities` declares the CogGym content the provider handles; image/video experiments need corresponding support. Optional `trial_ids` selects a labelled smoke subset; the registered benchmark uses the complete canonical selection.

The [CPU qualification example](../examples/coggym_toolbox.py) checks direct execution, recording, a seeded 50% ablation in one MLP layer, restoration, and call replay using cached SmolVLM-256M weights:

```bash
python examples/coggym_toolbox.py \
  --checkout /path/to/coggym \
  --model-path /path/to/SmolVLM-256M-Instruct/snapshot \
  --output-dir /path/to/new-results \
  --trials 3
```

This small check verifies tool behavior, not leaderboard performance. Stage both inputs first; the script never downloads, and with `HF_HUB_OFFLINE=1` a missing snapshot fails instead of fetching one:

```bash
HF_HUB_OFFLINE=0 huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct \
  --revision 7e3e67edbbed1bf9888184d9df282b700a323964 --local-dir /path/to/SmolVLM-256M-Instruct/snapshot
```

Before loading, the script checks the model directory against the pinned Hub listing for that commit (`SMOLVLM_MANIFEST` in the example). Each file the loader reads (weights, config, processor and tokenizer files) must exist with the listed size and hash, computed from the bytes on disk: the LFS SHA-256 for the weights and the Git blob SHA-1 for small files. The directory may be a `--local-dir` download as above, or a cached snapshot (`.../models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67e...`). A missing or changed file, any extra top-level file other than `README.md`, `.gitattributes`, `onnx/` and `.cache/`, a cache snapshot of another commit, or a download record naming another commit fails the run before the model loads. The `onnx/` files are not used or verified.

`qualification.json` is written whether the run passes or fails. It holds `status`, the first `error`, the verified model identity, a SHA-256 of the whole model directory, the CogGym and unified Git commits (with a tracked-edits flag), and package versions. Every check raises an explicit error, so `python -O` cannot skip it. Treat only `status: passed` as a qualification.

The same model, checked the same way, also drives the [native-environment check](../examples/native_environment_toolbox.py): it compares a direct `GridGameEnv` loop with recorded, ablated and restored `Experiment` runs on the same seeds. Restored and recorded actions and rewards must equal the direct loop; changes after ablation are reported, not required.

Neither run is a registered integration profile yet for CogGym: the profile runner stages only model and data caches, not a CogGym checkout, so the CogGym run stays a direct command. The native check is registered as the `native-environment-smolvlm` profile in `install/integration-tests.json`.

## Score identity and leaderboard policy

The registered profile uses five repetitions, temperature 1.0, and 512 tokens. Overrides add those settings to the resolved benchmark identifier so preliminary runs cannot share its score identity. These settings match the README example, not necessarily a published leaderboard row.

CogGym scores have no estimated ceiling. The benchmark and score declare `leaderboard_eligible=False`. Core's submission pipeline rejects excluded benchmarks before model loading, and excluded scores before database storage or S3 upload. This does not change local scoring or existing benchmarks that omit the flag. CogGym remains off Brain-Score leaderboards until a ceiling and reporting policy are established.


Match the public-set scope, checkpoint, prompts, trial IDs, sampling, repetitions, and media processing. The README's five-run example and the paper's ten-run description do not establish the settings of every leaderboard row. Report differences explicitly.

Matching the reference analyzer is separate from reproducing published scores. Hosted Qwen3.5-Flash corresponds to Qwen3.5-35B-A3B, but local serving still needs its own qualification.

Sources: [CogGym evaluator](https://github.com/lance-ying/coggym/tree/a1cd9df1118fec80eba7463de237d1497d77e041/evaluation), [paper](https://arxiv.org/html/2609.21259), [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
