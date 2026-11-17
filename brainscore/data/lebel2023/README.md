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
# The source pickle in its one place (or under $BRAINSCORE_DATA_HOME/sources/lebel2023)
SOURCE="$(python -c "from brainscore.data.packaging import source_folder; print(source_folder('lebel2023'))")"
mkdir -p "$SOURCE"
mv /path/to/UTS03-assembly.pkl "$SOURCE"/assembly_lebel_uts03.pkl
# The converted files of other settings go to ~/brainscore-data/lebel2023-converted
```

With the unified package installed, convert and load without a model:

```python
import brainscore

# Convert and load: one stimulus row per response row
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

Other settings convert in the loader. The hosted files are rebuilt from the pickle
with `python -m brainscore.data package lebel2023` (see Rebuild below); a rebuild
has not yet been compared with the registered sha1.

## Rebuild

### lana-atlas

- Source: the LanA language atlas (Lipkin et al. 2022), https://osf.io/kzwbh/  (doi:10.17605/OSF.IO/KZWBH); its own terms.
- Access: public, no account.
- Needs: a download, size not measured; no tools.
- Command: none: nothing to package.
- Writes: The atlas is read as published. Put the folder at `$BRAINSCORE_DATA_HOME/lana-atlas` (`~/brainscore-data/lana-atlas` by default), or put a link there to it.
- Check: nothing to compare.

### lebel2023

- Source: LeBel et al. 2023, OpenNeuro ds003020 (https://openneuro.org/datasets/ds003020), CC0, as the UTS03 assembly pickle of the LITCoder release (https://github.com/GT-LIT-Lab/litcoder_release).
- Access: not recorded here: the LITCoder release page states how to obtain the pickle; the recordings themselves are public (CC0).
- Needs: the unified install (numpy, pandas, xarray); about 3 GB of disk for the pickle and more than 3 GB of memory; time not measured.
- Command: `python -m brainscore.data package lebel2023` with the source in `~/brainscore-data/sources/lebel2023` (or `--source <folder>`): `assembly_lebel_uts03.pkl` (the UTS03 assembly pickle of the LITCoder release). Output (or `--output <new folder>`): each identical file at `$BRAINIO_HOME/<sha1>/<file name>` (`~/.brainio` by default), where the loader reads it; the build record in `$BRAINIO_HOME/packaged/lebel2023`.
- Writes: `assy_LeBel2023-UTS03.nc`, `stimulus_LeBel2023-UTS03.csv`, `stimulus_LeBel2023-UTS03-words.csv`, with the registered default settings.
- Check: sha1 of `assy_LeBel2023-UTS03.nc` is `e03fd25503004d82ea932e272dbbdcb867d99a92`; sha1 of `stimulus_LeBel2023-UTS03.csv` is `9ddf67c7f3f9702473248eb15a877377d11732e6`; sha1 of `stimulus_LeBel2023-UTS03-words.csv` is `ce3d1ba0dc7457d189547ca8cda2c87ebcc8d5c8`
