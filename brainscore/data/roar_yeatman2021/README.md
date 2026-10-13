# Load ROAR / Yeatman2021 data locally

This plugin downloads a versioned stimulus CSV, image archive and behavioral
assembly from Brain-Score storage, checks their hashes and caches them locally.
The versions and expected hashes are in `__init__.py`.

```python
from pathlib import Path

import brainscore

stimuli = brainscore.load_stimulus_set("Yeatman2021")
responses = brainscore.load_dataset("Yeatman2021")
print(len(stimuli), responses.sizes)
assert all(Path(path).exists() for path in stimuli.stimulus_paths.values())
```

The images contain words and pseudowords. The stimulus set also includes text
and ground-truth labels for the text benchmark. The assembly contains human
behavioral responses; its `correct` field supplies the ground-truth label,
not per-trial model accuracy.

There is no native-file builder or author request-ID resolver for this plugin.
The [dataset audit](../../../docs/dataset_preparation.md) requires confirmation
of access/redistribution terms for each asset, a fresh-cache loading check, and
documented size/runtime estimates before claiming general-release data support.
