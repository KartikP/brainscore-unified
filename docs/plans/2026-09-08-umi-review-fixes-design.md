# UMI review fixes: design and scope

User authorization: implement all six findings from the code review and leave an
auditable issue/fix/validation record. All four repositories are on
`unified-model-interface-v2`. Do not commit or push.

## Decisions

1. Algonauts prediction owns a fitted feature projection shared by training and
   held-out extraction. Keep the existing CV protocol, explicitly label its
   all-input SVD as transductive, and do not claim inductive CV scores.
2. Recorded subsets retain original per-layer unit indices. Functional and
   random selections use the same original population; reject ambiguous
   addresses instead of guessing.
3. Text StimulusSet rows are independent by default. An explicit `context_id`
   column groups ordered parts of a passage within a process call. Both native
   text wrappers and the permanent language adapter honor that declaration.
   Raw legacy digest_text calls retain their existing passage semantics.
4. Route recording targets by region and modality before reducing to layer
   paths. Region attribution uses modality as well as layer, including
   composite regions and shared layer names in separate towers.
5. Reset clears recorder, behavioral, adapter and policy-owned state. Selection
   probes restore the complete previous recording configuration. Benchmarks
   clean up their perturbations even if reading or partial installation fails.
6. The language adapter validates row correspondence and restores input
   presentation metadata on real assembly outputs.

These targeted contract repairs preserve process(), existing plugins and the
permanent adapters. A replacement interface would create unnecessary churn;
benchmark-specific repairs would leave the same defects available elsewhere.

## Verification

Use `/opt/anaconda3/envs/brainscore-unified-fresh/bin/python` and
`RESULTCACHING_DISABLE=1` throughout. Add regression tests with tiny synthetic
arrays and CPU models constructed from configuration, then run the unified
unit/integration tier and affected core, vision and language suites. No real
benchmark data, downloaded weights, or network access is needed.

The final issue/fix/test mapping and limitations will be recorded in
`docs/audits/2026-09-08-umi-review-fixes.md`.
