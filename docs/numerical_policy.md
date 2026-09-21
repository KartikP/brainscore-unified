# Native numerical policy

## Current evidence

The [clean-upstream comparison](qualification/2026-09-16-upstream-language.md)
found and repaired a token-cache difference in the v2 legacy language helper.
After repair, all four fixed GPT-2/Pereira cases agree exactly across upstream,
legacy, adapter, and native routes on the recorded macOS CPU profile. The budgets
below remain unchanged. Earlier drift measurements and Linux/GPU results describe
the previous helper; they do not qualify the repair on those platforms.

## Contract and scope

FP32 remains the production target. Exact compatibility applies to the legacy
adapter's activations. Its scalar raw and normalized scores retain a four-FP32-ULP
reduction allowance, without an absolute floor. Native policies never relax
these adapter checks.

`gpt2-pereira-cpu-fp32-v1` is an opt-in regression budget for **local GPT-2,
transformer.h.11, the fixed Pereira 243/384 linear and ridge cases, CPU FP32,
batch size one, full passage context without window eviction**. It is not a
general error bound for GPT-2, other models, reasoning traces, interventions,
robot policies, or arbitrary text. `historical-v1` retains the previous policy
and remains the CLI default. CUDA cannot select the CPU policy.

## Acceptance budget

| Quantity | CPU FP32 policy |
|---|---|
| Native activation | Every element within 32 effective FP32 ULPs; magnitude floor 64 |
| Native normalized score | Absolute difference at most 0.00001 |
| Native raw score | Absolute difference divided by the common ceiling at most 0.00001 |
| Mean signed score drift | At most 0.000005, separately for each pair of experiments and for normalized/raw scores |
| Stimulus and target ordering | Exact |
| Adapter activation/score | Existing exact/four-ULP requirements |
| Vision | Existing 0.000001 absolute native limits |

An effective ULP is FP32 spacing at `max(abs(reference), 64)`. Thus the
activation budget is 0.000244140625 below magnitude 128 and 0.0009765625 in
[256, 512). The floor makes this a mixed absolute/relative policy. It is an
empirical calibration, not a theorem about accumulated error. Thirty-two
spacings spend five fraction bits at that scale. The observed Pereira maximum
of 27 informed this calibration; these data are **not a holdout**.

The score budget is a declared reproducibility target: one part in 100,000 of
the normalized score scale (0.001 percentage points). It is not measured
neuroscientific uncertainty and does not guarantee identical rounded strings.
The mean signed budget reserves half that allowance for common direction drift.
This replaces the historical rejection of any same-sign nonzero change **only
in the opt-in CPU profile**. Equal clipped scores cannot hide raw-score drift.
Linear and ridge retain separate paired checks and ceilings.

The profile is useful for repeatable acceptance on these fixed cases. It cannot
certify an untested environment simply because the constants have a name. A
release report must identify checkpoint hashes, platform, numerical libraries,
thread counts and actual precision, and run all selected cases from cold result
caches. A software, checkpoint or device change requires a fresh qualification.

## Evidence and limits

On the macOS arm64 candidate stack (Python 3.11, Torch 2.13.0, Transformers
4.57.6, NumPy 1.26.4, scikit-learn 1.7.2, four CPU threads), the earlier full
FP32 run had exact adapter activations and scores on all eight release cases.
All four native vision cases were exact. Native Pereira activation maxima were
27/26 effective ULPs. Normalized score differences ranged from 0.0000008346 to
0.0000018035. All 627 token/position/mask audits passed without cache eviction.

A full FP64/eager language reference run produced exact exported activations
and scores across all three routes for all four cases. This uses FP64 for the
legacy reference too; it is not evidence that native FP64 exactly reproduces
legacy FP32. Wrappers export FP32 arrays to the benchmark metric.

Six newly authored text passages (66 recorded prefixes, up to 306 tokens) are
an explicit counterexample to extending the Pereira calibration. Their FP32
cached/full-prefix differences reached 269 effective ULPs with one thread and
540 with four threads. Their token semantics agreed. FP64/eager differences
were at most 3.41e-13 before export and zero after FP32 export. **The 32-ULP
Pereira profile does not accept these probes.** Raising a universal tolerance
to fit them would conceal rather than resolve the scope problem.

Tool authors should compare effect sizes against the numerical variability of
their own model, inputs and intervention. For small effects, use matched
execution paths, repeated controls, and a higher-precision diagnostic where
available. Window eviction is a separate semantic issue, not covered by this
numerical budget. Robotics action tolerances must be specified in the relevant
physical units; they do not inherit a language activation tolerance.

## Run

Stage the registered raw data and checkpoint files first. The runner refuses
missing-data downloads and records SHA-256 hashes of checkpoint contents.

```bash
RESULTCACHING_DISABLE=1 BRAINIO_HOME=/path/to/staged/brainio \
python -m brainscore.validation.run_parity \
  --resnet18 /path/to/resnet18.pth --gpt2 /path/to/gpt2 \
  --device cpu --threads 4 --policy gpt2-pereira-cpu-fp32-v1 \
  --out cpu-fp32.json
```

For the diagnostic reference, select `--device cpu --language-precision
float64-eager --policy historical-v1`. It cannot set `release_complete=true`.
For CUDA qualification use `--device cuda --policy historical-v1`; CUDA must be
available and TF32 is disabled. Results under the CPU profile cannot stand in
for CUDA results. `release_complete` describes this eight-case suite only,
not approval for the entire general production release.

## NVIDIA L4 FP32 profile

`gpt2-pereira-l4-fp32-v1` uses the same exact GPT-2/tokenizer inputs and four
fixed Pereira cases as the CPU profile. It requires `--device cuda`, FP32,
and an actual device name of `NVIDIA L4`. Other CUDA devices are rejected by
this profile; they need their own qualification evidence.

This profile **retains the historical 16 effective FP32 ULP activation budget**
with magnitude floor 64, and applies it consistently to linear and ridge.
The metric-input hashes must establish that these protocols see the same
activations for each experiment. The normalized and raw-scaled score budget
is 1e-5; the mean signed budget is 5e-6 within each experiment pair. These score
precision targets are shared with the CPU policy and were recorded before the
L4 language observations. Adapter checks and vision limits remain unchanged.

```bash
RESULTCACHING_DISABLE=1 BRAINIO_HOME=/path/to/staged/brainio \
python -m brainscore.validation.run_parity \
  --resnet18 /path/to/resnet18.pth --gpt2 /path/to/gpt2 \
  --device cuda --threads 4 --policy gpt2-pereira-l4-fp32-v1 \
  --out l4-fp32.json
```

The validation profile is an acceptance specification. Qualification requires
passing real-data case reports with the recorded Torch/CUDA/driver and numerical
library stack; merely selecting it does not establish production readiness.
Retain historical failures alongside the new profile's results. It is not a
universal bound for new prompts, interventions, models, devices or robotics.
