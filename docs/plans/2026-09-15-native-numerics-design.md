# Native numerical qualification

## Decision

Preserve the historical limits and model defaults. Following the user's explicit
choice to retain FP32 performance, add an opt-in, versioned GPT-2/Pereira CPU
regression budget with its limited scope documented. Add a CPU FP64/eager
diagnostic reference, retain failure evidence, and make placement/provenance
reproducible. Reference success cannot substitute for FP32 qualification.

## Evidence and alternatives

All eight legacy-adapter comparisons were exact on the macOS candidate stack.
The four native vision cases were also exact. The four native language cases
failed the existing limits: cached legacy and full-prefix native execution have
matching token, position, and causal-mask semantics but different FP32 results.
Pereira maxima reached 27 effective FP32 ULPs. Fresh text probes reached 269 at
one thread, so expanding a Pereira limit to 32 would not establish a general
error bound. The previous same-sign score guard also rejects arbitrarily small
common-sign differences. It remains unchanged pending a scientific policy decision.

Three options were considered:

1. Calibrate a wider FP32 budget. This requires a declared scientific precision
   target and broader evidence; a value fitted to Pereira is insufficient for
   a general toolbox. Adopt only a clearly scoped GPT-2/Pereira CPU profile;
   retain the fresh text failures as explicit counterexamples to broader use.
2. Add a high-precision reference mode. This permits reproducible diagnosis and
   checks whether divergence survives increased arithmetic precision. Select
   this option for qualification tooling; record that the legacy reference also
   uses FP64 and that exported activations remain FP32.
3. Make native execution reuse the legacy cache. This changes execution/state
   behavior and raises separate context-window semantics questions. It is not
   justified by the current evidence.

## Design

The standalone parity runner accepts an explicit device and language precision.
CPU reference mode constructs all language routes with FP64 parameters and eager
attention. CUDA requests fail before inference when CUDA is unavailable; they
cannot silently fall back to CPU. CPU runs suppress automatic MPS/CUDA selection
inside the standalone harness. Checkpoints and raw inputs remain local, and
persistent result caches are disabled. Reports include checkpoint hashes,
software versions, device, thread settings, precision, and route observations
on assertion failure. Reference-mode success cannot set the default release
completion flag. Historical numerical limits and adapter checks stay intact. The opt-in CPU
profile applies 32 effective ULPs to the four Pereira activation comparisons,
1e-5 normalized/raw-scaled score limits, and a 5e-6 mean signed limit within
each linear/ridge pair. See ../numerical_policy.md for rationale and limits.

## Validation

Run all four full Pereira cases in CPU reference mode, audit the paired guard,
and compare against the previously captured default-FP32 evidence. Probe new
text at one and four threads, preserving pre-export FP64 values. Test unavailable
CUDA, unsupported precision/device combinations, failure-report retention,
reference/default result separation, and existing scientific regression guards.
Prepare a checksummed source/wheel bundle and offline Linux/CUDA instructions.
Execution on that platform requires an existing machine or runner supplied by
the user; this work does not provision infrastructure.

## Release interpretation

A passing reference run is evidence about the reference configuration only.
FP32 outside the qualified scope, Linux/CUDA, published-score reproduction, and general release
approval remain independent gates. No claim of a universal FP32 tolerance or
new scientific accuracy guarantee follows from these experiments.
