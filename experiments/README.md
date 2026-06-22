# experiments/

One-off research runs kept for **provenance**, not maintenance.

Each subdirectory produced a specific finding (a sweep, a calibration, a
validation, a replication) whose result is recorded in `CLAUDE.md` /
the research notes. The code here is the driver that produced it — it
exercised library capabilities that now live under `brainscore/`, but it
is **not** itself reusable infrastructure and is **not** imported by the
library or tests.

- Not on the maintained path: no CI, no API stability, may have hardcoded
  EC2 paths and stale env assumptions.
- Never deleted (archive, don't `rm`): these are the record of how a number
  was obtained.
- If something here turns out to be reusable, it gets folded into
  `brainscore/` proper — at which point the driver becomes a thin caller or
  is retired.

Maintained, re-runnable drivers (data prep, caching, website assets, cost
bookkeeping) live in `scripts/`, not here.
