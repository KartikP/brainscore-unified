# External environments: qualification

October 29, 2026. This work strengthens the existing external-evaluator and native-session paths. It does not replace an evaluator's task loop or scoring.

## What changed

- LIBERO records require evidence that the evaluator finished. Interrupted calls and failed cleanup remain failures.
- LIBERO shutdown survives signals during Python cleanup. Saving and connection cleanup have a five-second deadline; a forced exit leaves an unfinished record incomplete.
- Remote sends and closes have deadlines. Original errors are retained, and uncertain actions are not retried automatically.
- Record validation streams the log. Optional byte budgets stop recording explicitly, and interrupted records can be inspected without accepting them as completed results.
- Long sessions can disable retained history and collected outputs. Output storage is checked before model execution.
- External-package conformance tests and portable qualification drivers cover both execution paths. Model qualification checks verify the pinned checkpoint files before loading.

## Candidate revisions

- Core: `b61263c5d47672bfa733424803dd720e1a5cc8b1`.
- Vision: `dc5553c72225f045a417f85c4e73d60f4196327c`.
- Language: `88c2e08712d7b53e20792aadc8511b9ac7c40fae`.
- Unified installed-package checks: `1489698fb1dc206538540ca7cb44f842f1001eaa`.
- Final unified shutdown correction: `b77f740`; the full offline suite and installed transport checks were rerun with this correction. It changes the standalone bridge and its tests, not installed package modules.

## Tests

| Check | Result | Scope |
| --- | --- | --- |
| Source unified offline suite | 1,596 passed; 17 optional skips; 63 slow tests deselected | Unit and fixture integration tests |
| Installed core | 821 passed | Required offline profile |
| Installed vision | 239 passed | Required offline profile |
| Installed language | 70 passed | Required offline profile |
| Installed unified | 1,597 passed; 16 optional skips; 63 slow tests deselected | Full offline profile; no unexpected skips or source-import violations |
| Installed LIBERO/OpenPI transport | 209 passed | Fixture policies; actual separate Python 3.8 evaluator and Python 3.11 bridge |
| Installed external partner example | 14 passed | Adapter/tool conformance, not a human usability trial |

The four packages were installed as wheels in a fresh Conda environment on macOS ARM64. Each coordinated test profile used a temporary home, empty result/model caches, offline mode and disabled cloud credentials. Imported package files were checked against the candidate wheels. Vision and language plugin catalogs and private-data scientific scoring were not run.

The shutdown regression fails on the previous bridge. Twelve Linux stress runs passed after the correction; additional fault tests cover blocked saving, save errors and repeated stop signals. An independent installed-package rerun passed all 209 transport tests.

The original full core suite had three failures and seven errors that also reproduced on unchanged v2: source-path/Conda handling and an endpoint fixture. They are separate from the required offline profile above.

### Long runs

Two independent processes each completed 1,000 and 10,000 steps for baseline, recording, scaled activity and restoration. Recording and restoration preserved actions exactly. Scaling changed the selected activity and actions. After an injected failure in the third episode, a fresh three-episode run restored baseline actions; no hooks or session history remained.

The fixed-weight CPU fixture used about 330–341 MB peak process memory. Recorded artifacts grew from 2.79 MB to 27.85 MB with step count. Allocation tracing showed a bounded CPython intern-table rebuild, reproduced without UMI; the live-object regression test detects per-step retention. These measurements are not performance budgets for trained models.

## Reproduce

From an environment containing the four candidate wheels and test dependencies:

```bash
python -I -m brainscore.validation.workspace \
  --root /path/to/sibling-repositories \
  --mode installed \
  --wheels /path/to/candidate-wheels \
  --out qualification/installed.json

env -u PYTHONPATH -u PYTHONHOME python -I -O unified/examples/external_environment_checks/run.py \
  --out qualification/workers \
  --steps 10000 \
  --workers 2
```

The workspace report records revisions, selected test IDs, skips and wheel identities. Use a new output directory for each run. See the [LIBERO guide](../../examples/libero/README.md) for the separately prepared evaluator, bridge and policy environments, and the [CogGym guide](../coggym.md) for its qualification command.

## Remaining qualification

- Trained LIBERO/OpenPI and real-weight SmolVLM native/CogGym runs at the final candidate. The drivers are prepared; the CPU fixtures do not establish trained-policy parity.
- A partner researcher integrating an actual task, such as ABC Pickup.
- Physical-robot safety, hard real-time control and automatic simulator-state recovery are outside this scope.

No new billable EC2 run, merge, release or deployment was performed. Earlier trained-policy results remain historical.
