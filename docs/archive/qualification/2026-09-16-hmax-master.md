---
title: UMI v2 - HMAX Master Comparison
date: 2026-09-16
status: scoped-compatibility-passed
tags:
  - brain-score
  - umi
  - production-readiness
---

# HMAX compatibility with current master

## Conclusion

The clean upstream HMAX runs match the UMI candidate's activation values, raw
scores, ceilings and normalized scores exactly for both MajajHong V4 and IT
on the recorded macOS CPU profile. The older published-score discrepancy also
exists on current upstream; it is not introduced by UMI in this comparison.

Original production logs were not needed to establish this. They would only
help reconstruct how the older website scores were produced. The earlier
readiness framing was too restrictive: missing historical logs should not by
themselves block UMI compatibility with current upstream.

## Full upstream benchmark results

| Region | Default mapped layer | Master raw score | Master normalized score | UMI legacy comparison | UMI adapter comparison |
| --- | --- | ---: | ---: | --- | --- |
| V4 | `c2_0` | 0.171058988692 | 0.032692956554 | Exact | Exact |
| IT | `c2_2` | 0.197480409501 | 0.047452644776 | Exact | Exact |

Each clean upstream benchmark performs fresh inference on all 2,560 private
images, plus its normal memory-preflight probe. Each selected layer has 400
features. Saved activation values match the corresponding layer of both prior
UMI legacy and adapter runs byte-for-byte. Image identities, image hashes and
row order match exactly. The comparison uses no relaxed numerical tolerance.

The older rounded published references are V4 0.161 and IT 0.101, corroborated
by the [V4 benchmark page](https://www.brain-score.org/benchmark/vision/307) and
[IT benchmark page](https://www.brain-score.org/benchmark/vision/308). This fresh
master run reproduces the same lower values recorded in the April experiment
and the recent UMI diagnostic. The cause of the older published gap remains
unknown; it is separate from the demonstrated upstream compatibility result.

## What was run

- Clean [vision master at `96108f73`](https://github.com/brain-score/vision/tree/96108f737df7f9aca16f93e144aafce72fb397f6),
  fetched September 16, 2026.
- Clean [core main at `b1604265`](https://github.com/brain-score/core/tree/b1604265d5bdb4a44c3f42d5472fd530ba7f2f1a).
- The normal `load_model('hmax')` factory and `score_benchmark` entry point,
  including memory preflight. HMAX uses its shipped V4 `c2_0` and IT `c2_2` map.
- No UMI or language package imports in either scoring worker. Every imported
  core/vision module location and file hash is recorded and verified against
  the clean upstream worktrees.
- Local macOS arm64 CPU, Python 3.11.15, FP32, batch size one and two compute
  threads per region. Torch 2.13.0, torchvision 0.28.0, NumPy 1.26.4, SciPy
  1.17.1 and xarray 2022.3.0 match the prior recorded CPU stack.
- Isolated scikit-learn 1.5.2 satisfies master's declared `<1.6` requirement.
  Prior candidate controls with the same version already reproduce the earlier
  1.7.2 scores exactly. No qualified environment was changed; dependencies were
  reused, so this does not constitute a fresh installation qualification.
- The same independently verified patch set, staged private dataset and image
  content. Downloads of unstaged inputs are refused.
- Fresh empty activation caches per region. Upstream HMAX temporarily overrides
  the global cache-disable request; observed forward counts confirm that every
  scoring image actually ran through the model. Cache hits cannot explain the
  matching outputs.
- Upstream source remains unmodified. The harness observes returned outputs,
  sets the explicit CPU/batch/thread configuration and restores the import-time
  legacy HTTPS override. It changes no weights, preprocessing, equations,
  mapped layers, benchmark definitions or score-normalization rules.

No AWS instance was started. Both local workers completed within the two-hour
controller bound. Full observations, completion receipts and audit are in
`artifacts/hmax-master-comparison-2026-09-16/` under the production workspace.

## Metadata and scope

This comparison verifies activation values and scientific scores under matched
stimulus and feature order. The prior diagnostic recorded all eight layers
with identifier `hmax-fixed-layers` and an IT recording label; the normal
master calls use identifier `hmax` and each actual target region. Whole metadata
structure identity is therefore not claimed across these differently configured
experiments. The earlier legacy index-level ordering limitation remains documented
in [the prior HMAX findings](2026-09-16-hmax.md).

These results cover the two default mapped HMAX CPU cases in this software and
data profile. They do not establish HMAX CUDA behavior, the automatic layer-map
fallback, full-data session execution, overlapping threaded calls, or reproduction
of the historical production environment. The runtime cleanup fixes retain their
separate 57 source checks and 13 diagnostic-wheel checks. The original qualified
candidate wheel set has not been replaced or requalified by this source run.

## Readiness correction

Mark HMAX's tested compatibility with current upstream as passed. Keep the old
published-score references and their mismatch records intact, but track that
historical reproducibility issue separately instead of calling it a demonstrated
UMI regression or requiring old logs to close this direct compatibility check.

The earlier HMAX findings remain valid measurements; this new upstream control
supersedes their recommendation to keep UMI compatibility blocked solely on the
unexplained historical gap. Any promise to reproduce the older website scores
would still require separate evidence.

General production release still requires a declared supported-profile matrix,
immutable release CI, real DROID episode/checkpoint integration, independent
author onboarding, named release/support owners, licensing and recovery checks.
The HMAX runtime follow-up is committed as `4bbcd626e62bc02df89c8206aecddc193233f7c8`.
Its tested file hashes are recorded in the diagnostic report. The original
16 candidate commits and qualified wheel identities are unchanged. Nothing
has been pushed.

Machine-readable comparison: [2026-09-16-hmax-master.json](2026-09-16-hmax-master.json).
