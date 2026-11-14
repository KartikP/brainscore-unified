# Build an integration outside Brain-Score

This package uses public UMI interfaces. Install the coordinated UMI packages
first, following the [installation guide](../../install/README.md).
Then, from the Unified repository:

```sh
python -m pip install ./examples/partner_tool
```

Once installed, run these commands from any directory. Use a new output folder
for each run. Neither example downloads a model or calls a paid service.

## Add an input and output

```sh
python -I -m umi_partner_tool --out ./channel-example
```

[The capability](src/umi_partner_tool/__init__.py) accepts a three-value signal
and returns its amplitude. It is a session capability: `supports_session` claims
the output channel and `participate` answers each signal. Each capability method
takes the model as its first argument, named `subject`. The example registers its
channels, attaches an observer, records the response, reads it back, and checks
reset. Copy the package
under your own distribution/module name and replace the calculation and validators.

## Combine tools across five cases

```sh
python -I -m umi_partner_tool.toolbox --out ./toolbox-examples
```

Every case records responses and activity, silences a layer, checks recovery,
and reruns saved inputs. `summary.json` reports the comparisons.

| Case | What it demonstrates |
| --- | --- |
| `vision` | Present image arrays and record activity and behavior. |
| `language` | Present text through the same session API. |
| `vlm` | Combine an image and text before producing a response. |
| `robotics` | Use `BrainScoreModel(policy=...)`; actions change the next observation. |
| `brain_behavior` | Carry state between inputs, then reset between conditions. |

Run one case with `--example vision` (or another name from the table).
These are small, synthetic fixtures. They do not establish trained-model
performance, brain similarity, or robot success. `neural:population` names the
toy model's units, not a biological brain region. No physical clock is claimed.

In [toolbox.py](src/umi_partner_tool/toolbox.py), `make_example` sets up the subject
and session; `run_example` shows the shared tool list. The first three cases and
the recurrent case implement the public `Subject` contract directly. The policy
case uses the ready-made `BrainScoreModel`.

Static cases compare complete saved responses exactly. Robotics compares action
arrays on saved observations; inference timing and execution receipts are kept
in the records but excluded from that comparison. Replay does not restart the
environment.

## Test your integration

```sh
python -I -m pytest ./examples/partner_tool/tests -q --import-mode=importlib
```

Run this from the Unified repository after installing the package. The tests
check installed imports, the five examples, invalid input, record/replay,
intervention recovery, and cleanup after failure. The environment conformance
tests also show how to check your own observation/action declarations.

These maintainer-built examples do not replace the [independent user trial](../../docs/archive/partner_trial.md).
