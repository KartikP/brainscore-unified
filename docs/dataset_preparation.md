# Local dataset preparation audit

**Source review: October 13, 2026.** This checks the preparation paths for the
release matrix's benchmark families and the new research integrations. It does
not certify access rights or a fresh full-data rebuild. Every benchmark included
in the final release scope must meet [the local preparation checks](local_data.md#release-requirement-users-can-prepare-the-data-locally).

| Dataset / integration | Existing path | What needs verification or improvement |
| --- | --- | --- |
| MajajHong2015 / FreemanZiemba2013 | Vision registers checksum-pinned assembly and stimulus downloads; packaging scripts are retained | Document the download/access route for each asset; verify a fresh authorized load and stimulus/response alignment |
| Pereira2018 | Language downloads checksum-pinned assemblies containing sentence metadata; packaging script is retained | Add a concise data README and fresh-cache validation; distinguish source sentences from model-generated features |
| ROAR / Yeatman2021 | Unified downloads versioned image stimuli and behavioral responses | Add a data README, confirm which assets may be redistributed, and verify fresh loading; a local native-file converter is not registered |
| Lahner2024 / BOLDMoments | Versioned downloads; scripts prepare time-resolved assemblies, events, motion and audio | Complete the local path: registered loaders still use hosted files, including for prepared time-resolved data. Document native inputs and validate both stimulus and response builds |
| LeBel2023 | An author-format pickle is converted locally into text stimuli and cortical responses, with context/trim-aware caching | Add a data README and data-only validation command; record source content identity and validate fresh-cache outputs. Atlas-dependent variants need the atlas too |
| Algonauts2025 | DataLad/native HDF5 and movie/transcript files become local NetCDF assemblies and stimulus CSVs | Add a data README and shared-builder entry; validate paths, splits and incomplete builds. Held-out NaN response placeholders are not measured human responses |
| EWoK | Shared `prepare` command accepts native tables, the protected paper archive, or approved Hugging Face access; both registries load the build | Best existing shared-builder example. Recheck the release candidate against the canonical 4,372-item build; preserve the documented difference from the paper's item count |
| CogGym | Pinned author checkout retains trials, media and human reference responses; benchmark loading checks selected inputs | Keep its native format. Verify completeness and access for every supported experiment; do not claim that all catalog data or retrieval IDs are supported |
| LAION-fMRI (legacy example) | Vision has an author request-ID path, assembly rebuild and semantic comparison against hosted references | Reuse the workflow, not the scripts unchanged: pin upstream source, review the automatic license-marker step, verify native-file-only use and loading locally built assemblies |
| DROID / LIBERO | Episode/checkpoint staging and native simulator/evaluator setup, rather than a conventional static neural assembly | Record episode IDs, access/setup, observation/action schemas and reference checks for the supported robotics profile; do not require conversion merely for uniformity |

## Complete the release check

For each supported dataset, assign its data maintainer and retain:

1. The README with source version, permissions, native input format and exact commands.
2. A reproducible authorized download or local build of **both** stimuli and targets.
3. Validation results for IDs, alignment, dimensions, coordinates, preprocessing,
   exclusions and expected content; document permitted numerical differences.
4. A fresh-directory load through the benchmark's normal entry point, without
   developer caches, code edits, uploading restricted data or loading a model.

Prioritize the final #6 release scope. Research examples use the same checklist,
without turning their mere registration into a support commitment. Unavailable
source assets require an explicit scope decision.

## Source evidence

- Unified: `brainscore/data/{ewok,coggym,lebel2023,algonauts2025,lahner2024,roar_yeatman2021}/`.
- Vision: `brainscore_vision/data/{majajhong2015,freemanziemba2013,laion_fmri}/`.
- Language: `brainscore_language/data/pereira2018/`.
- Tests: `test_ewok.py`, `test_local_data.py`, `test_lebel2023_data.py`,
  `test_algonauts_heldout.py`, `test_coggym_benchmark.py` and domain data tests.

At the start of this audit, a file inventory found 41 vision, 7 language and 6 unified data-plugin folders.
Of those, 4, 0 and 2 contain a README; 30, 5 and 3 contain candidate preparation
files. These are inventory counts, not failure counts: documentation may live
elsewhere, and the presence of a script does not establish that it runs.

This documentation pass adds local preparation/readme instructions for LeBel,
Algonauts, Lahner and ROAR. It does not mark their outstanding checks complete.
