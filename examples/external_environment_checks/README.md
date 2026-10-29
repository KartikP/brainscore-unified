# Check long experiments locally

This CPU check uses a fixed-weight one-layer policy and a counter environment.
It tests plumbing and tools, not trained-policy performance or benchmark replication.

```bash
python run.py --out new-results --steps 1000 --workers 2
```

Each worker gets a separate process, subject, environment and output directory.
Each runs baseline, recording, scaled activity and restoration. It then reuses
the subject across three episodes, injects an environment failure during the
third episode with scaling active, and checks a fresh three-episode run. The check requires:

- Exact action agreement between baseline, recording and restoration.
- Changed actions and the expected recorded activation values under scaling.
- All steps completed, hooks removed, environments closed, no retained session history.
- Event logs and their payload assets within the configured 128 MiB reservation limit (manifests and other experiment files are separate).

`report.json` reports wall time with allocation tracing enabled, Python allocation
peak, process peak RSS, artifact size and latency percentiles over the last 1,024
policy calls. Results describe this small workload; they are not GPU, trained-policy,
robot-control or throughput guarantees. `workers.json` collects worker results.
A failed check writes a failed `report.json` and exits with an error. Checks
remain active under `python -O`. These checks do not promise cleanup after a
forced process kill.

Use a new output directory for every run. Model and result caches are not used.

Memory and latency are measurements, not acceptance budgets. History retention
is checked explicitly. The fixture policy is stateless; its failure check targets
environment stepping and hook cleanup. RNG isolation follows from separate
processes and is not measured by this deterministic fixture. Source imports and
installed distribution metadata are reported separately.

`python_peak_bytes` can jump by several MB once in a long run. That is CPython reallocating
its process-wide interned-string table (`pathlib` interns path parts, including recorded
array file names), which `tracemalloc` reports as a new block. It is not retained per step:
the test `test_recorded_instrumented_episode_keeps_no_per_step_objects` checks that the
number of live objects after a full garbage collection does not grow with steps. Provenance
reports `source_commit` only for a clean, tracked checkout; otherwise it is null and
`git_head` with `source_dirty` describe the checkout.
