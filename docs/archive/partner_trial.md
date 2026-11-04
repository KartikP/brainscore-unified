# Try building your first UMI tool

This trial checks whether a researcher can build an integration using the guide
without changing Brain-Score's contracts. It has not yet been completed by an
unfamiliar author.

## The task

In your own package, build a tool that records one model measurement during an
experiment. Save the result, read it back without running the model, and verify
that removing the tool leaves the model usable.

Choose a measurement you need: response length, action latency, one layer's
activity, or a domain-specific output. Start with the
[tool guide](tool_authoring.md) and [example package](../examples/partner_tool/README.md).
Use a local model or fixture that does not require paid API access.

## What to record

- Time to install, first result, and completed integration.
- Commands used, package revisions, and environment.
- Every missing instruction or place you needed help.
- Whether any core contract or Brain-Score source edit was needed.
- Saved output, replayed measurement, and cleanup result.

Stop the clock when waiting for data or credentials. Keep those delays separate
from time spent understanding the interface. Do not silently repair the guide
during the trial; capture the problem first.

## Success

The author produces a working external package using documented public APIs.
The measurement survives record/replay, cleanup restores normal operation, and
no contract edit is needed. A helper can clarify questions, but log each answer.
A maintainer-built example is useful evidence and does not replace this trial.
