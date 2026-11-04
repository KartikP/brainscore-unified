# LeBel2023 story-listening fMRI (UTS03)

With the default settings no preparation is needed: `brainscore.load_dataset("LeBel2023-UTS03")`
and `brainscore.load_stimulus_set("LeBel2023-UTS03")` / `("LeBel2023-UTS03-words")` fetch the
hosted files from `brainscore-storage/brainscore-language/data/LeBel2023-UTS03/` and check them
against their sha1 (`hosted.py`). The recordings are from OpenNeuro ds003020 (LeBel et al., 2023,
*Scientific Data*), released under CC0; the files were converted from the LITCoder UTS03 assembly
with `data_packaging/package_hosted.py`.

## Other settings: build locally

A context window other than 5 TRs (`context_window_tr=`) or 32 words (`context_words=`) is built
from the source. Obtain the UTS03 assembly pickle through the
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
[dataset audit](../../../docs/archive/dataset_preparation.md).
