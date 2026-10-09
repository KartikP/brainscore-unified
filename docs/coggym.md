# Use UMI tools with CogGym

CogGym runs the experiment and scores its responses. UMI observes the model calls and attaches recording or intervention tools to the model.

`CogGymRunner` calls CogGym's existing prompt builder, trial selector, evaluator, parser, and analyzer. It does not translate the experiment into a different task loop.

## Select a registered experiment

All 23 public experiments are registered as `CogGym.<Study>.<experiment>`. For example, `CogGym.Hu2023Fine.exp1` selects the text experiment. Registration provides access; it does not mean that every model or modality has been qualified.

With your provider and response-trace subject prepared:

```python
import brainscore

benchmark = brainscore.load_benchmark(
    "CogGym.Hu2023Fine.exp1",
    checkout="/path/to/coggym",  # Use the pinned checkout below.
    reset=provider.reset,  # Reset provider history and sampling for each repeat.
    repetitions=1,  # A preliminary pass, not a full leaderboard replication.
    temperature=1.0,
    max_tokens=8192,
    output_dir="runs/coggym-score",  # A new directory; existing runs are preserved.
)
score = brainscore.score(subject, benchmark)
```

The result is CogGym's **raw Pearson R²**, with coverage and configuration in `score.attrs`. No human ceiling is applied. Undefined correlations raise an error and preserve the trial records. Without `output_dir`, records remain in a new temporary directory whose path is returned in `score.attrs["run_directory"]`.

To attach tools, use `benchmark.protocol(model=subject.identifier)` in the `Experiment` example below instead of `runner.protocol(...)`. It uses the configured provider reset and returns the same `Score`. A provider without state can explicitly use `reset=lambda repetition: None`; UMI does not assume that reset is unnecessary.

Set `BRAINSCORE_COGGYM_CHECKOUT` to omit `checkout`. You can list names without loading data or a model:

```python
names = sorted(
    name for name in brainscore.benchmark_registry
    if name.startswith("CogGym.")
)
```

The registry covers 11 text, 6 image and 6 video experiments. Media experiments need a provider that actually handles the supplied media. Loading validates the reference revision and prompts; it does not download data automatically. The defaults are one repetition, temperature 1.0 and an 8,192-token limit. Match the reference settings explicitly before claiming replication.

## Prepare the reference evaluator

Use a separate experiment environment. Install the four UMI packages first, then clone and pin CogGym:

```bash
git clone https://github.com/lance-ying/coggym.git
git -C coggym checkout a1cd9df1118fec80eba7463de237d1497d77e041
python -m pip install -r coggym/evaluation/requirements.txt
```

CogGym imports the Google SDK even for local providers. Installing it does not require an API key or make API calls.

```python
from brainscore.harnesses.coggym import CogGymRunner

runner = CogGymRunner(
    "coggym",
    experiment="Hu2023Fine/exp1",
    model="your-provider-model-id",
    temperature=1.0,
    max_tokens=512,
    repetitions=1,
)
```

Construction checks the pinned source, canonical trial selection, and prompts before model loading. The adapter rejects tracked edits to the evaluator or data. By default, the provider is declared to support text. For a multimodal provider, declare `modalities=["text", "image", "video"]`; this declaration does not implement image or video processing.

The public manifest limits the available experiments. An optional `trial_ids=[...]` selects a smoke subset in canonical order; its records explicitly identify that reduced scope.

## Connect a model and tools

Supply a provider implementing CogGym's `complete_with_metadata(system, messages, model, temperature, max_tokens)`. It returns a dictionary with `text`, `reasoning`, and `token_usage`. Use `None` for unavailable reasoning or token counts. Preserve all supplied text and media.

```python
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore.experiments import (
    Experiment,
    RecordInputsOutputs,
    RecordReasoning,
    RecordActivity,
    TorchInstrumentation,
)

subject = build_trace_subject(
    "my-coggym-model",
    provider=lambda request: provider.complete_with_metadata(**request),
    parse=str,  # CogGym interprets the answer with its own parser.
    provenance={"checkpoint": "exact revision", "precision": "float32"},
)

result = Experiment(
    subject=subject,
    protocol=runner.protocol(reset=provider.reset),
    tools=[
        RecordInputsOutputs(),  # Save calls and tool measurements.
        RecordReasoning(),  # Save reasoning only when the provider exposes it.
        RecordActivity(["encoder"]),  # Use a real layer path from your model.
    ],
    instrumentation=TorchInstrumentation(provider.model),
    output_dir="runs/coggym-recorded",
).run()
```

Here `provider` and its `model` are supplied by your integration. `provider.reset(repetition)` receives a one-based repetition number. It must reset conversation state and sampling as appropriate, without replacing the instrumented model or removing its hooks. Store the seed policy, checkpoint, precision, decoding, and media-processing settings in provenance. UMI does not infer these settings.

The adapter passes only CogGym's provider arguments to the model. Human scoring targets remain in the evaluator. Each call uses the reference runner's independent trial prompt. Records include the experiment, repetition, attempt index, and IDs whose reference prompts match that request. Duplicate prompts retain multiple matching IDs rather than an invented unique association. These indices are not physical timestamps.

## Check recording and interventions

1. Call `runner.run(provider, output_dir=..., reset=provider.reset)` for a direct reference run.
2. Run the `Experiment` with recording under the same model and sampling settings.
3. Compare actual requests, raw responses, parsed answers, and scores.
4. Add `Ablate([selection])` before `RecordActivity` in a separate experiment.
5. Remove the intervention and confirm that the original outputs return under controlled settings.

An ablation must name its scope. For example, zeroing half the intermediate units of one MLP layer is not the same as removing half the model's neurons. Use the shared `Selection` type; its indices address the last output axis.

For dampening, use `ScaleActivity(targets, factor=0.5)` with `TorchInstrumentation`. For a model exposing `model.language_model.layers`, this selects every fifth full block:

```python
from brainscore.experiments import ScaleActivity

targets = [
    f"model.language_model.layers.{index}"
    for index in range(4, len(provider.model.model.language_model.layers), 5)
]
dampen = ScaleActivity(targets, factor=0.5)  # Halve outputs at blocks 5, 10, 15, ...
```

Add `dampen` before the after-intervention `RecordActivity` tool. This scales the whole block's output, including its residual stream. It does not randomize or replace weights. Use the same prompts and generation settings for baseline and intervention. Compare scores on the shared scorable questions, and report invalid or truncated responses separately. Different internal activity does not imply worse benchmark performance.

The [CPU example](../examples/coggym_toolbox.py) uses a cached, pretrained SmolVLM-256M model in text mode. It compares direct execution, recording, a seeded 50% ablation in one MLP layer, restoration, and actual call replay:

```bash
python examples/coggym_toolbox.py \
  --checkout /path/to/coggym \
  --model-path /path/to/SmolVLM-256M-Instruct/snapshot \
  --output-dir /path/to/new-results \
  --trials 3
```

The model must already be cached. This is a small tool qualification, not a CogGym leaderboard run.

## Read the results

| Location | Contents |
| --- | --- |
| `experiment.json` | UMI setup, completion status, and artifact provenance |
| `inputs_outputs/` | Actual model requests, responses, and recorded activity |
| `reasoning/` | Provider-exposed reasoning or an explicit unavailable marker |
| `coggym/run-001.json` | Reference trial results and summary, with added scope and coverage metadata |
| `coggym/analysis.json` | Reference analyzer output, including undefined results and coverage |

CogGym catches provider errors per trial. The adapter saves those results and raises if trials failed, were omitted, or returned out of order. Unparseable responses remain visible as unscorable trials; they are not silently replaced. Check coverage alongside any score. CogGym's analyzer can omit experiments with insufficient usable responses or undefined correlation.

Trial results are saved before summary aggregation. If native scoring fails, the experiment remains failed and the saved artifact identifies the scoring error. Calls and activity are still available for inspection; a scoring failure is not a zero score or a completed benchmark.

`replay_calls` resends saved requests to a model. Viewing saved responses requires no inference. Neither operation reruns an interactive environment. See the [tool guide](experiment_toolbox.md#replay-saved-inputs).

## Comparing with the leaderboard

Match the public-set scope, checkpoint, prompts, trial IDs, sampling, repetition count, and media processing. The public README's five-run example and the paper's ten-run description are not interchangeable guarantees of a particular leaderboard row. Preserve the reference analyzer and report any differences from the published setup explicitly.

The public analyzer is the authority for this adapter's scores. Reproducing its behavior is separate from reproducing the published leaderboard values. Hosted Qwen3.5-Flash corresponds to Qwen3.5-35B-A3B, but a local deployment still needs its own checkpoint and serving qualification.

Sources: [CogGym evaluator](https://github.com/lance-ying/coggym/tree/a1cd9df1118fec80eba7463de237d1497d77e041/evaluation), [paper](https://arxiv.org/html/2609.21259), [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
