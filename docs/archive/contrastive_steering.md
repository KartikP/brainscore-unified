# Fit and apply contrastive steering

This tool learns `direction = mean(positive) - mean(negative)` on explicitly
supplied discovery activity, then applies `activity + strength * direction`.
Positive and negative strengths add opposite offsets; zero preserves activity.
The direction is a descriptive contrast, not evidence that its labels identify
a unique causal mechanism or that behavior will change monotonically with dose.

The method relates to [Contrastive Activation Addition (Rimsky et al., 2024)](https://aclanthology.org/2024.acl-long.828/).
That work uses paired residual-stream differences and applies them after the
prompt. This tool supplies the mean-difference and addition primitives. It does
not supply prompt construction, pairing, token alignment or a generation-token
scheduler. Balanced paired examples have the same mean difference; unmatched
groups can encode nuisance differences. Design those controls in the protocol.

Run the self-contained CPU example without model downloads:

```bash
RESULTCACHING_DISABLE=1 python examples/contrastive_steering.py --out /tmp/new-steering-demo
```

It saves one direction, reloads two signed doses, observes both through before/
after probes in one experiment, and checks exact restoration on a fixed
synthetic encoder. This qualifies integration on that fixture, not real-model
or human effects.

```python
from brainscore.tools import load_tool

direction = load_tool("contrastive-steering", target="language", strength=0.5).fit(
    discovery_activity, discovery_labels,
    positive="truthful", negative="untruthful", normalize=False,
    provenance={"checkpoint": "exact revision", "dataset": "discovery-v1"},
)
direction.save("direction.json")
steer = load_tool("contrastive-steering", artifact="direction.json", strength=-0.5)
shifted_activity = steer.transform(evaluation_activity)
```

`fit` returns an independent tool and accepts `(presentation, neuroid)` activity
with one nonempty string label per example. Both named groups must occur, and
other labels fail explicitly. Choose any pooling before fitting. Preserve the
target's `layer` coordinate and original integer feature addresses (`unit_index`,
`neuron_number_in_layer`, or `neuroid_num`). Coordinate checks do not verify
checkpoint identity; supply that provenance explicitly.

Fitting uses float64 group means. `normalize=False` preserves their scale;
`normalize=True` gives a unit L2 vector and rejects a zero direction. Raw zero
directions are valid identity additions. Labels, counts, normalization and the
direction are fixed in the saved artifact. No extra fitter is required.

## Compose inside a protocol or benchmark

```python
tools = [
    load_tool("record-io"),
    load_tool("record-activity", targets=["language"], name="before", when="before"),
    load_tool("contrastive-steering", artifact="direction.json", strength=0.5,
              name="positive", conditions=["positive"]),
    load_tool("contrastive-steering", artifact="direction.json", strength=-0.5,
              name="negative", conditions=["negative"]),
    load_tool("record-activity", targets=["language"], name="after"),
]
```

Pass this collection to `Experiment` with the declared conditions, or use
`attach_tools` around caller-owned sessions. Direct scopes have condition
`default`; leave filters unset or use that value. Each instance needs a unique
name. Conditions and trials never implicitly fit a direction.

The backend must declare `transform`, currently local `TorchInstrumentation`.
Bind functional population names explicitly to modules. The same coordinates,
`Selection.axis` and `expected_shape` used for recording select the injection
site. Each selected batch/token/spatial position receives the same vector,
including prompt and padding positions if they pass through that site. Repeated
forwards reuse it. Use uncached execution; a scope without a target forward
fails. Hooks detach on completion, errors or interruption.

Interventions compose in list order. Steering before scaling scales the added
offset; scaling before steering does not. Multiple steering directions add,
subject to floating-point rounding. A later patch can replace a steered value.
Weights and the subject contract stay unchanged.

`strength` must be a finite number. Saving retains its default dose; loading
allows an explicit override without refitting. Each attachment snapshots the
dose, so changing `tool.strength` affects the next scope. For a protocol dose
sweep, use separately named instances with condition filters. The artifact's
effective checksum includes the dose, while fitting provenance stays the same.

Versioned, checksummed JSON stores the direction, target, default dose, fitting
fingerprint, settings and provenance without raw discovery examples or pickle.
Files are never overwritten. Loading and offline application import neither
PyTorch, transformers, sklearn nor a fitting package. Offline results are
`NeuroidAssembly` objects preserving coordinates and floating dtype.

Live application uses the actual activation dtype/device and preserves tensor
shape and tuple tails. Offline calculation uses float64 before casting back;
live low-precision arithmetic can differ. `tool_artifact` events retain the
effective parameters; application events retain actual shape, dtype, device,
package version and signed dose. Nonfinite outputs fail and detach hooks.
Use held-out evaluation, opposite-sign, zero-dose and unrelated-task controls
when testing a scientific hypothesis.

[Pretrained-model qualification](qualification/2026-11-02-interpretability-tools.md)
checks signed and zero doses on GPT-2 and CLIP text against native float32
addition, including composition with scaling and exact restoration.
