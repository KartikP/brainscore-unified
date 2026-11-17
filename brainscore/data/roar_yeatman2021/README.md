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

No source is known, so the files cannot be rebuilt yet (see Rebuild below).

## Rebuild

### yeatman2021

- Source: not stated. The benchmark cites the ROAR lexical decision task of Yeatman et al. 2021 and the protocol of Honarmand et al. 2026; the files were put on the Brain-Score bucket in unified commit e1085ba0 (2026-07-02) with no packaging script. No address and no licence are recorded.
- Access: not known for the source; the hosted files download with no account.
- Needs: not known.
- Command: none yet. No public source is known, so no rebuild exists. A maintainer needs: the source of the images and the trials, and its terms, from the ROAR authors (Yeatman lab); then a script in `data_packaging/` that writes the three files, and a comparison with the registered sha1.
- Writes: `stimulus_Yeatman2021.csv`, `stimulus_Yeatman2021.zip` (word and pseudoword images), `assy_Yeatman2021.nc` (human lexical decisions). Use the hosted files: `brainscore.load_dataset` fetches them.
- Check: sha1 of `stimulus_Yeatman2021.csv` is `04fbe35ba4384cf8ba5d152961d21f278a77e32a`; sha1 of `stimulus_Yeatman2021.zip` is `0adacf476500f8f8a4b0a6d34f8688b22b583fcd`; sha1 of `assy_Yeatman2021.nc` is `fee134858a68ac4bd113f6b6ec79f4f8722db1b8`
