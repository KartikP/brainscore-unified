# Fitted interpretation tools

The approved scope adds LEACE first, then contrastive steering, with one tested
commit per method. Existing tool constructors and scores retain their behavior.

## Ownership and interface

Tools fit only on caller-supplied discovery assemblies. A fitted tool can be
saved, loaded, applied to another assembly, or attached to a protocol/benchmark.
No fitting happens implicitly during evaluation. Target layer and original unit
addresses travel with the fitted artifact; there is no private-model discovery.

The existing subject/session contract stays unchanged. Add one optional
`transform` operation to PyTorch instrumentation. It applies a callable to
selected floating activity, preserves shape/dtype/device and tuple tails, owns
its hooks, and removes them after normal exit or failure. Each active scope
must execute its selected sites; cached-only responses cannot count as measured
interventions. Multiple forwards are supported for fixed fitted transforms.

Use common artifact/lifecycle support for both methods, alongside the existing
ablation, scaling, patching and recording tools. Keep numerical fitting and
application in method-specific modules. Tool names, ordering, conditions and
trials retain their current meanings. Backend support is explicit.

## Scientific and data boundaries

LEACE delegates fitting to the reference `concept-erasure` library. Its
covariance shrinkage, SVD threshold and trace constraint are recorded settings;
they are not guarantees of erasing every representation of a concept.
Contrastive steering uses an explicit difference of group means, with
optional declared normalization and a separately configurable application dose.
Save the default dose with the artifact and allow an explicit loading override.
Snapshot that dose at attachment so parameters and recorded events agree for
the entire scope. Distinct named instances support protocol dose sweeps.

Require original feature addresses and explicit labels. Preserve assembly
coordinates and source provenance. Artifacts contain parameters, target,
settings, fitting-data fingerprint and dependency versions; no pickle or
executable callbacks are serialized. Fitted parameters do not migrate between
model revisions automatically. Runtime records retain actual parameters and
application shape/dtype/device; callers still supply checkpoint provenance.

## Verification

Check LEACE against the reference implementation on held-out examples. Check
steering against independently calculated means and expected signed effects.
Verify artifact integrity, subsets/axes, gradients, tuple outputs, repeated
forwards, multiple-tool ordering, inactive conditions, restoration and failure
cleanup. Supply runnable CPU fixtures. Run the full offline unified suite before
each method commit. Real-model scientific qualification remains separate.
