# Inspect and intervene in a robotics policy

Use the same `RecordActivity`, `Ablate`, and `Selection` tools used for other models. The instrumentation provider connects them to the policy's internals.

| Policy | Provider | Scope |
| --- | --- | --- |
| Local PyTorch | `TorchInstrumentation(network)` | The network's named modules |
| Local OpenPI/JAX | `OpenPIInstrumentation(policy, checkpoint=...)` | Three sites in the pinned Pi0/Pi0.5 sampler; **feature branch** |
| Remote OpenPI/JAX | `RemoteOpenPIInstrumentation(client)` | Same sites through `OpenPIToolServer`; **feature branch** |
| Other JAX policies | Custom provider | No general JAX layer-hook API is included |
| Actions-only service | `RecordInputsOutputs` | Requests and actions; no internal access |

See [notebook 19](../notebooks/19_openpi_experiment_toolbox.ipynb) for saved trained-policy measurements and a short live-client example.

## OpenPI: choose what to measure

A diffusion policy starts with noise and refines an action chunk over several **denoising steps**. These are internal computations, not simulator steps.

| Target | What it contains | Last-axis units |
| --- | --- | --- |
| `embed_suffix` | Action, state and time embeddings entering the transformer | Embedding features |
| `PaliGemma.llm.suffix` | Transformer output for the action sequence | Hidden features |
| `action_out_proj` | Predicted change used to refine the actions | Action dimensions, before integration and output transforms |

Select all units with a target name, or selected units with `Selection(layer=..., indices=[...])`. Arrays keep their batch and sequence axes. `steps=[0, 2]` limits recording and ablation to those zero-based denoising iterations; omit it for all iterations. Ablation sets selected values to zero at that site. It does not directly zero the final robot commands. Recording selects units after transferring the site output to the host, so selecting fewer units reduces saved data but not that transfer.

```python
from brainscore.experiments import (
    OpenPIInstrumentation, RecordInputsOutputs, RecordActivity, Ablate,
)
from brainscore_core.events import Selection

# Use the actual policy that will answer inference requests.
instrumentation = OpenPIInstrumentation(
    policy,
    checkpoint='pi05_libero:sha256:<checkpoint-and-normalization-digest>',
    steps=[0, 2],
)
target = Selection(layer='action_out_proj', indices=[0])
tools = [
    RecordInputsOutputs(),
    RecordActivity([target], when='before', name='before'),
    Ablate([target]),
    RecordActivity([target], name='after'),
]
```

Pass `instrumentation` and `tools` to an `Experiment`. Use `CallableProtocol` when an external evaluator owns inference calls, resets and scoring. For a native session, use `SessionProtocol` and choose ablation conditions/trials as usual. Observation transforms, action chunks and evaluator rules stay under OpenPI/LIBERO's control.

Each measurement includes the array, selected indices, denoising iteration, diffusion time and original dtype. The experiment links it to the current call/trial. The manifest identifies the checkpoint, sampler, JAX version and devices. Bfloat16 recordings are stored as float32 for portability; the model computation retains its original dtype.

## Remote policy

On the policy machine, run [serve_tools.py](../examples/libero/serve_tools.py) in the pinned OpenPI runtime with UMI available:

```bash
python examples/libero/serve_tools.py \
  --checkpoint /path/to/pi05_libero \
  --config pi05_libero \
  --steps 0 2 \
  --port 8001
```

The launcher hashes local checkpoint and normalization files. It binds to loopback. For another machine, forward that port through SSH; this server has no built-in authentication or TLS.

On the experiment machine:

```python
from brainscore.experiments import (
    Experiment, CallableProtocol, OpenPIPolicyClient,
    RemoteOpenPIInstrumentation,
)

with OpenPIPolicyClient('ws://127.0.0.1:8001') as policy:
    experiment = Experiment(
        subject=policy,
        protocol=CallableProtocol(
            'policy-evaluation',
            evaluate,  # Your evaluator calls policy.infer(observation).
            methods=['infer'],
        ),
        tools=tools,
        instrumentation=RemoteOpenPIInstrumentation(policy),
        output_dir='runs/policy-evaluation',
    )
    result = experiment.run()
```

The client returns the normal inference result. Activity travels separately, with request and inference indices, and is delivered to the experiment's tools. This protocol requires the UMI tool server; it is not compatible with an unchanged OpenPI websocket server. An existing `LiberoChunkPolicy` can wrap this client without changing its action-chunk handling.

**Scope and cleanup:** configuration acknowledgments validate the requested tools. The server attaches them only during one locked inference call and removes them before replying, including on failure. Idle connections expire after five minutes by default. A disconnect does not cancel computation already running, but another call cannot inherit its intervention. The client never retries inference automatically. Use a dedicated policy server per experiment: interleaving clients still advances the shared RNG, even though their tools are isolated.

## Compatibility and qualification

Use OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`, JAX `0.5.3` and Flax `0.10.2`. OpenPI has no public hook for these values. This adapter verifies the sampler source hash and inserts probes at three assignments before compilation. The upstream loop, RNG, transforms and integration arithmetic are retained. Changed sampler source is rejected. Weights must remain fixed after policy construction, matching OpenPI's compiled-state behavior.

CPU tests use the real sampler with small, untrained layers. Trained qualification on an NVIDIA L4 passed exact recording checks on ten inputs at each of the three sites. Projection ablation/restoration also passed locally and through the remote client. A paired LIBERO Spatial smoke run completed 10/10 tasks with recording off and on; all 223 calls matched exactly. See the [qualification report](qualification/2026-10-06-openpi-tools.md) for compiler controls, the selection fix, and limits.

For a trained checkpoint, [qualify_tools.py](../examples/libero/qualify_tools.py) runs baseline, recording, ablation and restored inference on saved requests with matched RNG state. It saves four experiment records and `qualification.json`; recording and restored actions must match exactly. A failed comparison must be investigated, not hidden by widening tolerances afterward.

```bash
python examples/libero/qualify_tools.py \
  --checkpoint /path/to/pi05_libero \
  --record /path/to/completed/call-record \
  --out /path/to/new/qualification-directory \
  --limit 10 \
  --steps 0 2
```

**Still to qualify:** broader policies, GPUs, selections and full task suites. The isolated OpenPI server runtime is not the general UMI scoring environment. Restart the server and reuse GPU autotuning results between matched simulator runs; neither closing a tool scope nor reconnecting resets OpenPI's RNG. Physical-robot control and hardware safety are outside this integration.

The [remote qualification script](../examples/libero/qualify_remote_tools.py) tests the client/server path. The [evaluator bridge](../examples/libero/serve_tool_bridge.py) accepts the official evaluator's websocket calls and applies baseline, recording or ablation tools through the UMI model adapter.

## Add another backend

Implement `describe()`, `validate(operation, targets)`, `record(targets, receive)` and `ablate(targets)`. Both context managers must restore their changes after errors. `Selection.indices` selects the last output axis. Deliver supported arrays/mappings to `receive(target, value)`. Researchers keep using the same experiment tools; no new `Subject` method is needed.

[LIBERO integration](robotics_benchmark_integration.md) · [Experiment toolbox](experiment_toolbox.md)
