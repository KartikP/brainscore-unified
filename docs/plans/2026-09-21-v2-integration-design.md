# Coordinated productionization merge into v2

Integrate the productionization work into the four v2 branches without publishing
packages or claiming general availability. Preserve exact candidate peer versions:
install all four local checkouts or wheels in one pip operation. A standalone
candidate install cannot obtain its unpublished peers from PyPI.

Include the pending language full-context repair, DROID provider, tests, and
qualification documentation. The language repair also requires diagnostic replay
to respect recorded full-context calls and passage cache resets.

Record immutable core, vision, and language commits in the unified repository.
The pull-request workflow builds all four wheels, installs them together, checks
installed tools away from source, and runs all four offline source suites on Linux
and macOS. The shared CPU integration constraints identify the numerical stack.
The remote v2 heads have no divergence from the productionization branches.

Merge only after the coordinated integration checks pass, with unified last.
Preserve existing developer worktrees rather than switching or resetting them.
Full repaired-source Linux/CUDA scientific qualification, trained-policy evidence,
and the other production gates remain separate from this development-branch
merge. The prior CUDA reports must not be relabeled as evidence for the repair.
Kartik retains ownership of package publication and production service rollout.
