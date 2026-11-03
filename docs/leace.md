# Fit and apply LEACE concept erasure

LEACE learns an affine map for removing linearly accessible information about
specified categorical labels. Fitting delegates to the
[reference concept-erasure implementation](https://github.com/EleutherAI/concept-erasure).
The tool does not establish that every representation of a concept has been
removed or that a behavioral change is specific to that concept.

Install the fitting dependency with `pip install 'brainscore[interpretability]'`
in your coordinated UMI environment. Loading a learner or applying a saved map
offline does not import the fitter, PyTorch, or transformers.

Run a self-contained CPU fixture with:

```bash
RESULTCACHING_DISABLE=1 python examples/leace.py --out /tmp/new-leace-demo
```

It reuses a fitted eraser and two probes on a fixed synthetic text encoder,
compares baseline/erased/restored conditions and checks restoration. It is an
integration demonstration, not a trained-model or human validation.

```python
from brainscore.tools import load_tool

eraser = load_tool("leace", target="language").fit(
    discovery_activity, discovery_labels,
    provenance={"checkpoint": "exact revision", "dataset": "discovery-v1"},
)
eraser.save("eraser.json")
loaded = load_tool("leace", artifact="eraser.json", name="erase_gender")
erased_activity = loaded.transform(evaluation_activity)
```

`fit` returns an independent fitted tool. Supply `(presentation, neuroid)`
activity with the target's `layer` coordinate and original integer unit
addresses (`unit_index`, `neuron_number_in_layer`, or `neuroid_num`). Labels
are nonempty strings, with at least two classes; they are one-hot encoded.
Use a target matching the recorded layer address and bind instrumentation to
that address. A functional selection may supply this address without a benchmark
hard-coding model layers.

Fitting uses float64. Defaults match the reference library: `shrinkage=True`,
`constrain_cov_trace=True`, `svd_tol=0.01`, and an affine bias. The first three
can be configured when creating the learner. SVD thresholding, covariance
estimation and finite precision affect residual information. Fit settings are
stored in the artifact and cannot be overridden when loading it.

LEACE fitting stores a covariance matrix and uses quadratic memory in the
selected feature count. Choose your recorded population explicitly; the tool
does not silently apply PCA or remove units to fit a memory budget.

## Apply inside an experiment or benchmark

```python
tools = [
    load_tool("record-io"),
    load_tool("record-activity", targets=["language"], name="before", when="before"),
    load_tool("leace", artifact="eraser.json", conditions=["erased"]),
    load_tool("record-activity", targets=["language"], name="after"),
]
```

Pass these to `Experiment` with a protocol declaring `erased`, or use
`attach_tools` around caller-owned sessions. A direct scope uses condition
`default`, so omit filters or use that value. The backend must support
`transform`; currently this is the local `TorchInstrumentation` provider.
The subject contract and existing scoring are unchanged.

The saved map applies at every selected batch/token/spatial position without
pooling or masking. `Selection.axis` and `expected_shape` retain their usual
meanings. Repeated forwards, including generation steps, can reuse the map.
This does not provide a token scheduler. Hooks accept a floating tensor or
the first tensor of a tuple, preserving remaining entries.

Live application uses the activation's device and dtype. Offline application
returns a `NeuroidAssembly` with its coordinates and floating dtype preserved.
NumPy/CPU and low-precision backend results may differ numerically; only baseline
restoration is checked for exact equality. Mixed interventions use explicit list
order. Removing one scope removes only its hooks.

Artifacts use checksummed, versioned JSON and record parameters, fitting labels'
class set, settings, feature addresses, data fingerprint and package versions.
Supply checkpoint/dataset provenance explicitly; coordinate checks do not verify
checkpoint identity. `tool_artifact` and intervention application events preserve
the fitted map and actual shape/dtype/device in the shared record. No raw fitting
examples are embedded. Existing artifacts are not overwritten.

Use real, uncached forwards when measuring interventions. A scope that never
executes its target fails explicitly. Invalid maps and nonfinite results fail
and detach hooks. Check held-out probe behavior and unrelated-task controls;
reference-library parity and fixture checks are not human causal validation.

[Pretrained-model qualification](qualification/2026-11-02-interpretability-tools.md)
also checks held-out and live transforms on GPT-2 and CLIP text, including
artifact reuse, composition and exact restoration on CPU float32.
