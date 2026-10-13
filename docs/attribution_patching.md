# Find and verify influential units

Use **attribution patching** to estimate which unit replacements affect a model's
answer preference. Then use **activation patching** to measure the replacement's
actual effect. A large activity difference alone does not establish influence.

| Component | Role |
| --- | --- |
| `TorchInstrumentation.attribute` | Compare two inputs and estimate each selected unit's effect using gradients |
| `Selection` | Name the layer and optional unit indices, as for recording and ablation |
| `PatchActivity` | Replace selected outputs with recorded values during an experiment |
| `RecordInputsOutputs` | Save requests, responses, patch references and analysis events |

These operations require a local, differentiable PyTorch forward path. They do
not change `Subject`, the benchmark's metric, or model weights. Existing
OpenPI/JAX and remote providers do not implement these two operations.

## Estimate effects

Prepare two matched inputs: a **reference** and a **comparison**. For example,
change the operation in an arithmetic question while preserving token alignment.
Define one scalar objective, such as the log-probability of answer A minus that
of answer B. Use the same objective and answer candidates for both inputs.

The callbacks below return the model's raw outputs. They are backend analysis
calls, not ordinary detached UMI predictions. `answer_preference` must return a
scalar tensor connected to the comparison forward pass.

```python
from brainscore.experiments import TorchInstrumentation

model.eval()  # Disable training behavior; gradients remain available.
instrumentation = TorchInstrumentation(model)
analysis = instrumentation.attribute(
    ["transformer.h.5.mlp.act"],  # A GPT-2 layer path; use your model's path.
    reference_call=lambda: model(**reference_inputs),
    comparison_call=lambda: model(**comparison_inputs),
    objective=answer_preference,
    location=(0, -1),  # Batch 0, final token; retain its unit axis.
)
unit_scores = analysis["attributions"]["transformer.h.5.mlp.act"]
```

The estimate is `(reference activity − comparison activity) × comparison
gradient`. By default it is divided by the reference-minus-comparison objective
difference. Set `normalize=False` to retain the objective's original units.
Near-zero denominators, mismatched shapes/dtypes and detached objectives fail.
A target that ran but has no gradient connection receives explicit zero scores
and appears in `disconnected_targets`.

`location` indexes every axis except the final unit axis. Omit it to retain all
positions; decide explicitly how to aggregate them before ranking units. The
method does not align tokens, identify reasoning steps or sum effects across
layers. Contributions at different layers overlap and are not additive.

## Verify with exact patches

Freeze unit selection using discovery examples. On independent validation
examples, collect reference activity and patch the same unit IDs into their
matched comparison calls. Select units again only for a new discovery study.

```python
from brainscore.experiments import PatchActivity, RecordInputsOutputs
from brainscore_core.events import Selection

# Put these in Experiment(tools=...) with the same instrumentation provider.
tools = [
    RecordInputsOutputs(),
    PatchActivity(
        [Selection("transformer.h.5.mlp.act", [12, 47])],
        reference=analysis["reference"],
    ),
]
```

The reference includes values, shapes, dtypes, unit IDs and coordinates. Patching
can use a subset of recorded units. It copies outputs rather than overwriting
the caller's tensor, and removes its hooks on success or failure. References
are saved in the run; the plan includes their content hash.

Each patch scope permits **one invocation per target**. Missing or repeated
invocations fail. Use one paired forward per trial, or separate `Experiment`s.
An autoregressive `generate()` loop invokes layers repeatedly and is not a
supported exact-patch scope here. Ordinary `Ablate` and `ScaleActivity` still
support repeated calls, including CogGym evaluation.

For direct verification without an experiment:

```python
with instrumentation.patch(
    [Selection("transformer.h.5.mlp.act", [12, 47])],
    reference=analysis["reference"],
):
    patched_output = model(**comparison_inputs)
```

## Run the trained-model tool check

The [GPT-2 example](../examples/attribution_patching.py) uses cached weights,
CPU inference, 16 selected units, three matched random selections and a restored
baseline. It saves normal experiment records. It makes no downloads.

```bash
python examples/attribution_patching.py \
  --model-path /path/to/cached/gpt2 \
  --output-dir /path/to/new-run
```

This checks next-token preference on one prompt pair. Selection and evaluation
use that same pair: it is not evidence of a general circuit or held-out success.

## Transfer to CogGym

1. Establish baseline performance on a separate pilot set.
2. Choose paired localizer examples for a specific ability; retain unrelated
   tasks as controls. Keep discovery and evaluation examples separate.
3. Estimate effects, freeze selections, then verify them with exact patches.
4. Run CogGym with the frozen selections through `Ablate` or `ScaleActivity`.
   Keep its evaluator and scoring unchanged; compare matched random selections,
   unrelated tasks, completion rates and restoration.

Use actual answers and answer preferences together. Changed preferences need
not flip the final answer; decreased human-response R² is not automatically
decreased task accuracy. Null results remain valid outcomes.

Method reference: [de Varda et al., attribution patching and causal validation,
Appendices B–C](https://arxiv.org/pdf/2609.04463). This implementation supplies
mechanisms, not a replication of that paper's dataset or full protocol. It uses
two forward calls and a backward calculation; it does not train the model or
accumulate parameter gradients. Callbacks must preserve model state, and no
concurrent use of the model is supported.
