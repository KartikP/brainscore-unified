# Inspect and intervene in a robotics policy

A policy chooses actions from observations. Its implementation determines how we access internal activity. Robotics uses the same `RecordActivity`, `Ablate`, and `Selection` tools as other domains.

## Choose the backend connection

| Policy implementation | Instrumentation | Available now? |
| --- | --- | --- |
| Local PyTorch network | `TorchInstrumentation` on the actual network used during inference | Yes; no robotics-specific replacement is needed |
| Local JAX or another backend | An instrumentation provider for that backend | Extension interface exists; a JAX implementation is not included |
| Remote policy server | Instrumentation on the server plus a client adapter | Existing LIBERO bridge records calls/actions; internal capture and intervention need server changes |
| Actions-only API | Input/output recording | Yes; internal activity and ablation are unavailable unless the service exposes them |

For a local PyTorch policy:

```python
# This must be the same network the policy uses for inference.
instrumentation = TorchInstrumentation(policy_network)

# layer_path is a real module path in that network.
tools = [
    RecordInputsOutputs(),
    RecordActivity([layer_path], when='before', name='before'),
    Ablate([layer_path], conditions=['silenced']),
    RecordActivity([layer_path], name='after'),
]
```

Supply these to an `Experiment` with a session protocol and declared conditions. If an external evaluator owns the trials, use `CallableProtocol` and separate experiments for baseline/intervention conditions. Keep the evaluator's observation transforms, action chunk schedule, resets, and success rules unchanged.

Recording or changing returned actions is different from recording or changing internal policy activity. Name the measurement/intervention accordingly.

## Add a backend provider

Implement the existing instrumentation methods in your own package:

| Method | Responsibility |
| --- | --- |
| `describe()` | Identify the backend, target names and supported operations for the run record |
| `validate(operation, targets)` | Reject unavailable targets or operations before inference |
| `record(targets, receive)` | Return a context manager that sends selected activity to `receive(target, value)` |
| `ablate(targets)` | Return a context manager that temporarily zeros selected outputs |

Both context managers must undo their own changes, including after partial attachment failures. Use `Selection.layer` for a stable target name; match the documented last-axis meaning of `Selection.indices`. Returned measurements must use supported record payloads such as arrays and mappings.

The researcher continues to use `Experiment`, `RecordActivity`, and `Ablate`. Only the instrumentation provider changes. No new `Subject` method or robotics-only tool family is needed.

## Recommended next integration: OpenPI/JAX

The trained LIBERO example calls OpenPI in a separate policy server. To add internal tools:

1. Expose named intermediate values in the policy's inference computation. Recording must work with its compiled execution; a Python callback around `infer()` only sees inputs and actions.
2. Add explicit intervention sites at those values. Specify which step, selected units, and action-chunk computation each intervention affects.
3. Scope server recording/interventions to an isolated experiment. Return measurements with request identifiers, step/chunk indices, checkpoint, device, and precision. Keep them out of the action array consumed by LIBERO.
4. Implement the client instrumentation provider. Attach/detach operations need acknowledgments; disconnects must not leave an intervention active for a later run. Use server-side expiry or termination of the dedicated worker as a cleanup backstop.
5. Qualify against direct inference: recording alone preserves actions; a targeted intervention changes the intended activity; cleanup restores the baseline under matched state/RNG settings. Test interrupted calls, unsupported selections, and reset behavior before running paired simulator trials.

This is an implementation plan, not a claim that remote JAX instrumentation already works. The existing trained-policy LIBERO evidence establishes a different boundary: requests and actions through UMI. Physical-robot control and hardware safety remain outside this integration.

[LIBERO integration](robotics_benchmark_integration.md) · [Experiment toolbox](experiment_toolbox.md)
