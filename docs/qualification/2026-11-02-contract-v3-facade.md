# Contract v3 facade: regression qualification

November 2, 2026. **Scoped regression review complete; Malania excluded by maintainer decision.**

The candidate combines the Contract v3 simplification with a permanent facade that
runs legacy benchmark calls through `Subject.interact(session)`. This assessment
checks whether that change preserves existing measurements and scores. It is not
qualification of every supported model or environment.

## Revisions

The tested runtime came from `feat/contract-v3-facade`. The handoff branch is
`feat/contract-v3-qualified` in all four repositories; it preserves those runtime
revisions and adds qualification documentation in unified.

| Repository | Candidate tip | Runtime revision pinned by unified | Clean-v2 control |
| --- | --- | --- | --- |
| core | `bdda00955f8cba784f9d88c7d1bd708d8d95d265` | `15dcc3c6cbfea16565052abd2663b8de4bf1b203` | `20f9a749fbbdbebadc2beef73072df1dcb318085` |
| vision | `fb487671ee794055c16d511f22d6783fcc94afb8` | `eca24202bb7b188c1115d3b140b96abc0fa848c7` | `b5454d7e0c4c8cf7fc529b9c993753a6bc003450` |
| language | `02aff759089ee17bad5f0d93da437cb90c5e8fb5` | `62e2dcfe17daa81ecdf939f6e075d9edd33389ca` | `85ecb1d751d9d36a6938a86bfd3f85c4bbfda315` |
| unified | `b8880276242296c6857363777b4bd12a02ab5213` | Peer revisions above | `f08a427bf4936526f30f064e6899b9d1e4826cdc` |

Core and domain candidate tips add CI pin changes above the runtime revisions.
The unified revision above identifies the tested runtime; the commit adding this
report changes documentation only.

## Results

| Check | Result | Scope |
| --- | --- | --- |
| Independent focused tests | 258 passed | core 138; vision 59; language 46; unified 15 |
| Real-model matrix | 15 candidate cases executed; 14 exact candidate/base score matches | Malania assessed separately below |
| Candidate slow parity | 5 passed; no failures or skips | Six benchmark variants; legacy, adapted and native routes |
| Clean-v2 slow parity | 5 passed; no failures or skips | Same six variants and routes; approximately 3 hours 39 minutes |
| Saved candidate/base parity records | Exact equality for all six variants | Scores, raw scores, metric-input hashes and activation-delta records |
| Candidate regression-baseline pytest | 11 passed; no failures or skips; 2 deselected | Selected slow tier; approximately 1 hour 46 minutes |

The baseline pytest assertions use their configured numerical tolerances. The
candidate/base equality comparisons are separate, exact checks. Focused tests are
not a fresh full-suite run. HMAX matches clean v2 exactly in all three matrix cases
here; that does not explain historical production differences.

## Real-model matrix

The table compares full saved float values, independently of the M3 script's
`1e-4` pass threshold. A missing base value means the base adapter raised before
scoring, not a numerical mismatch.

| Model | Benchmark | Candidate | Clean v2 | Delta |
| --- | --- | ---: | ---: | ---: |
| alexnet | MajajHong2015.IT-pls | 0.41747309548349826 | 0.41747309548349826 | 0.0 |
| alexnet | MajajHong2015.V4-pls | 0.49191369552332914 | 0.49191369552332914 | 0.0 |
| alexnet | Allen2022_fmri_surface.IT-rdm | 0.380785274537124 | 0.380785274537124 | 0.0 |
| alexnet | Rajalingham2018-i2n | 0.37019382595706984 | 0.37019382595706984 | 0.0 |
| alexnet | Ferguson2024color-value_delta | 0.7185095193325057 | 0.7185095193325057 | 0.0 |
| alexnet | Malania2007.short2-threshold_elevation | 0.0 | Adapter error | Not comparable |
| hmax | MajajHong2015.IT-pls | 0.04745275573430901 | 0.04745275573430901 | 0.0 |
| hmax | MajajHong2015.V4-pls | 0.03265216069352088 | 0.03265216069352088 | 0.0 |
| hmax | Rajalingham2018-i2n | -0.046380463698292435 | -0.046380463698292435 | 0.0 |
| pixels | MajajHong2015.IT-pls | 0.012640346863273495 | 0.012640346863273495 | 0.0 |
| pixels | MajajHong2015.V4-pls | 0.06117406569325342 | 0.06117406569325342 | 0.0 |
| compact_vgg19_V4 | MajajHong2015.IT-pls | 0.25413549232253335 | 0.25413549232253335 | 0.0 |
| compact_vgg19_V4 | MajajHong2015.V4-pls | 0.638880653822547 | 0.638880653822547 | 0.0 |
| compact_vgg19_V4 | Rajalingham2018-i2n | 0.029997578701006493 | 0.029997578701006493 | 0.0 |
| compact_vgg19_V4 | Ferguson2024color-value_delta | 0.5789276099179909 | 0.5789276099179909 | 0.0 |

Eight pairs had production references; seven did not. The differences from those
production references also occur on clean v2. No dependency, ceiling or historical
cause is assigned by this comparison. Missing references were not replaced with
candidate-generated baselines.

The audit-only production query was corrected to use `Model.name`, filter by domain
and non-null scores, and retain benchmark version and score timestamp. The original
query referred to the nonexistent `Model.identifier` and could return no references
while exiting successfully.

## Slow parity

All three routes return the score below on both candidate and clean v2. Their raw
scores and recorded metric-input hashes also match exactly. Maximum activation
difference is zero for every route.

| Benchmark | Score | Candidate minus base |
| --- | ---: | ---: |
| MajajHong2015.IT-pls-unified | 0.38977962292028295 | 0.0 |
| MajajHong2015.V4-pls-unified | 0.5124659940384678 | 0.0 |
| MajajHong2015public.IT-pls-unified | 0.38771015493162714 | 0.0 |
| MajajHong2015public.V4-pls-unified | 0.5054701253162602 | 0.0 |
| Pereira2018.243sentences-linear-unified | 0.8712618939410148 | 0.0 |
| Pereira2018.384sentences-linear-unified | 0.8292698976046283 | 0.0 |

The Pereira variants are historical **linear** benchmarks used for compatibility
checks. These results do not qualify the current published ridge benchmarks.
The root M3 matrix has 15 vision pairs including Malania and no enabled language
pairs; language evidence here comes from the Pereira parity tests.

## Malania limitation

Clean v2's adapter raises on `require_variance`. A separate reference therefore
runs the raw legacy model directly. Direct base, direct candidate and candidate
facade runs all complete, and both measured response arrays match exactly across
the three routes. Observed fitting settings, trial counts and variance settings
are preserved.

All three return `0.0` after the benchmark reports a psychometric threshold-fit
failure caused by low accuracy. Matching fallback scores do not establish a valid
threshold fit.

The serialized call comparison remains false. Presentation-coordinate tuples have
different ordering between processes, and the facade forwards the trial count
positionally where the direct call uses a keyword. The probe did not save named
MultiIndex levels, so coordinate fidelity remains unverified. Resolve that evidence
gap remains unverified. Do not discard coordinate checks or classify the difference
as harmless without verifying it.

Kartik accepted Malania as outside the public benchmark scope for this review and
closed this qualification work on November 2. It is an explicit exclusion, not a
passing test. Revisit this evidence before making a support claim for Malania.

## Environment and execution

The checks ran on Linux EC2 `g6.4xlarge` workers in `us-east-2`, with NVIDIA L4
hardware. The candidate parity suite used a separate worker with the same pinned
environment and explicit source-import checks.

| Package | Version |
| --- | --- |
| Python | 3.11 |
| numpy | 1.26.4 |
| xarray | 2022.3.0 |
| scikit-learn | 1.7.2 |
| transformers | 4.57.6 |
| torch | 2.6.0 |
| scipy | 1.17.1 |
| pandas | 2.3.3 |

The environment was built from pinned requirements. Source datasets and model
weights were reused; result caches were isolated per matrix process or disabled for
parity. This was controlled regression qualification, not a download-cold user trial.
The M3 helper disables memory preflight; focused memory tests cover the probe
separately.

The initial clean-v2 parity attempt was interrupted by a shutdown deadline. It was
rerun to completion. The final run has JUnit counts, per-variant JSON and a captured
pytest exit status. The user removed the final wall-clock cutoff; evidence was
exported before the completion monitor stopped the worker. Both workers are stopped.

## Evidence and follow-up

[Structured results](2026-11-02-contract-v3-facade.json) retain the numerical
comparisons and test counts. Full logs, JUnit, package freezes, source revisions and
response arrays are retained privately by the maintainer; benchmark data and
credentials are not included here.

This scoped regression review is complete by maintainer decision. No regression
was observed in the 14 comparable matrix cases or six parity variants. Malania is
excluded with the limitations above. No source was merged or package released by
this qualification.

Next: open coordinated review PRs against `unified-model-interface-v2`, retain
the accepted exclusion in their descriptions, and verify final merged peer pins
with CI. Any runtime fix needs affected checks repeated against the corresponding
clean-v2 control. The broader release matrix remains separate roadmap work.
