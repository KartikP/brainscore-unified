# Prepare LeBel2023 data locally

Obtain the UTS03 assembly pickle through the
[LITCoder source](https://github.com/GT-LIT-Lab/litcoder_release) and its access
instructions. The loader expects that author's `SimpleNeuroidAssembly` layout,
including story words, their onset times, TR times and cortical responses.
It does not accept arbitrary LeBel files or raw scans.

```bash
export BRAINSCORE_LEBEL_PICKLE=/path/to/UTS03-assembly.pkl
export BRAINSCORE_LEBEL_CACHE=/path/to/new-cache
```

With the unified package installed, convert and load without a model:

```python
import brainscore

stimuli = brainscore.load_stimulus_set("LeBel2023-UTS03")
responses = brainscore.load_dataset("LeBel2023-UTS03")
print(len(stimuli), responses.sizes)
assert len(stimuli) == responses.sizes["presentation"]
```

The source pickle is approximately 3 GB; conversion also needs room for cached
outputs and more RAM than the source file size. Full conversion runtime and peak
memory have not been qualified for this release.

Each stimulus row contains the text preceding one fMRI sample. The default
context spans five TRs. Response alignment trims ten TRs from the start and five
from the end of each story; changing that convention changes the benchmark.
The language-mask benchmark additionally needs the LanA atlas.

Conversion happens in the loader, not through `brainscore.data prepare`.
A source-content manifest and a fresh-build validation record remain part of the
[dataset audit](../../../docs/dataset_preparation.md).
