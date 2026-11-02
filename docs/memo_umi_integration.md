# memo and Brain-Score UMI integration report

## Scope and dependency

This example branches from unified `b8880276242296c6857363777b4bd12a02ab5213`
(`feat/contract-v3-facade`). That facade candidate is still awaiting real-model
regression qualification and is not merged. The integration depends on its
v3 `Subject.interact(session)` contract and the pinned sibling revisions below;
it does not qualify that candidate or alter the qualification work.

| Repository | Revision used |
|---|---|
| unified base | `b8880276242296c6857363777b4bd12a02ab5213` |
| core (from `install/peer-revisions.json`) | `15dcc3c6cbfea16565052abd2663b8de4bf1b203` |
| vision (from `install/peer-revisions.json`) | `eca24202bb7b188c1115d3b140b96abc0fa848c7` |
| language (from `install/peer-revisions.json`) | `62e2dcfe17daa81ecdf939f6e075d9edd33389ca` |
| memo source reviewed | `d8aac635ee53483e8e0311b85f6898ef37710a26` (`kach/memo` main) |

The memo runtime was installed from the pinned `memo-lang==1.3.3` distribution;
the memo source revision above is the upstream implementation inspected against
the paper and the actual worker API. Its
`pyproject.toml` declares `requires-python = ">=3.12"` and `jax` as its runtime
dependency. The isolated worker environment pins `memo-lang==1.3.3`,
`jax==0.12.0.1`, and `jaxlib==0.12.0.1`. Unified pins Python 3.11, so the only
memo/JAX code runs in a small persistent Python 3.12+ JSON-lines worker. The
worker forces the JAX CPU backend and accepts batched utterances for native
calls. The subject responds to each utterance before asking the session for its
next input, so reactive sessions can advance on the response. Each transport
request has a finite deadline covering nonblocking writes and reads through the
complete response line. Partial replies and blocked writes trigger forced child
cleanup; worker errors are reported explicitly. `MemoSubject.reset()` shuts the
worker down. Memo/JAX are not core or unified dependencies.

## Paper task and fidelity contract

The paper's §2 example presents a green square, green circle, and pink circle;
the utterance set is green, pink, square, round. The memo implementation follows
the paper's RSA `@memo` model and the repository's `demo/demo-rsa.py` structure:
uniform referent prior, truthful utterances, literal depth 0, recursive listener
depth 1, and beta 1.0. The output ordering is explicitly
`green square`, `green circle`, `pink circle`; the input ordering is
`green`, `pink`, `square`, `round`.

Section 3 describes memo's agent-specific frames of mind and array-program
backend. The adapter leaves both mechanisms in memo: it calls the repository's
`@memo` model and transports only the resulting posterior probability rows.
Those rows are behavioral choices, not neural activity or reasoning traces.

The paper prints these tables rounded to two decimal places:

| Listener | Utterance | P(green square) | P(green circle) | P(pink circle) |
|---|---|---:|---:|---:|
| literal, depth 0 | green | 0.50 | 0.50 | 0.00 |
|  | pink | 0.00 | 0.00 | 1.00 |
|  | square | 1.00 | 0.00 | 0.00 |
|  | round | 0.00 | 0.50 | 0.50 |
| recursive, depth 1 | green | 0.43 | 0.57 | 0.00 |
|  | pink | 0.00 | 0.00 | 1.00 |
|  | square | 1.00 | 0.00 | 0.00 |
|  | round | 0.00 | 0.57 | 0.43 |

These are the paper's displayed values, not a substitute for a fresh memo run.
The checked-in [`results/summary.json`](../examples/memo_umi/results/summary.json)
contains the full precision direct memo and UMI tables. For each depth the run
required exact array equality between the direct worker call and UMI response,
and exact output equality after replay. It did not use a numeric tolerance to
compare those same-setting executions.

At beta 1.0, the depth-1 green-utterance probabilities were exactly
`[0.43022581934928894, 0.5697741508483887, 0.0]` in choice order. Depth 0 gives
`[0.5, 0.5, 0.0]`; both conditions' recorded inputs matched exactly.

## UMI integration and recording

The RSA subject and its worker live in `brainscore/models/memo_rsa/`. The
experiment and notebook live in `examples/memo_umi/`. The example constructs
the subject explicitly with its worker interpreter; importing the model package
does not start a worker or import memo/JAX into the host.

`MemoSubject` is a plain `Subject` with declared `instruction` and `text` inputs,
both required, and the existing `behavior` output. The task instruction describes
the three-object scene; each text event is one utterance. The response is a
probability vector in a `behavior` `StreamEvent`, with the existing `label_set`
and `stimulus_id` metadata. UMI's buffered session collects these into a typed
`BehavioralAssembly` whose coordinates preserve utterance and referent labels.
Nested imagined speaker/listener agents remain internal to memo.

The example uses the existing `Experiment`, `SessionProtocol`,
`RecordInputsOutputs`, `replay_sessions`, and `compare_outputs` tools. Input replay
loads and re-presents recorded events to the subject, so memo runs again; the
recorded responses are not fed back as outputs. It records settings and runtime
provenance on the experiment and each response, including model depth, beta,
worker and host Python versions, memo and JAX versions, JAX backend, and the UMI
base and sibling revisions plus the reviewed memo source revision. Each response
also records SHA-256 checksums of the subject, transport, worker, and memo model
source files; host NumPy version; output dtype; actual inference dtype; JAXLIB
version; JAX x64 and matrix precision settings; and CPU device details. Source
checksums identify this implementation independently of its Git commit. Depth 0
versus depth 1 uses identical input events and is described only as a
model-configuration comparison.

The follow-up [comparison notebook](../examples/memo_umi/comparison.ipynb)
adds an independent plain Python RSA implementation at the user's request.
Both models use the same denotations, uniform prior, and success-probability
speaker utility. Each preserves its native predictions exactly through UMI and
saved-input replay, with identical experimental inputs. Python's binary64 and
memo's default JAX binary32 outputs are compared by reporting their measured
difference, without a cross-implementation tolerance test. The notebook's
15 cells include executed probability tables, source code, provenance, and
replay results. Its events use `t_ms=None` because the task has no physical clock.
All 10 code cells executed successfully with the isolated Python 3.11 UMI and
Python 3.12 memo runtimes. Both implementations preserved native predictions
and replay outputs exactly. At depth 1, their maximum absolute difference was
`1.772241542408625e-08`, reported without a tolerance-based pass/fail claim.
The focused memo and Python reference tests passed (38 tests); the reference
checks include the literal table, an analytical depth-1 posterior, unambiguous
choices, symmetry, and zero speaker rationality.

## Environment and commands

The source tree is isolated at `/private/tmp/umi-memo-demo` on branch
`feat/memo-umi-demo`. The memo worker used a new `/private/tmp` virtual
environment with Python 3.12.4, memo-lang 1.3.3, JAX/JAXLIB 0.12.0.1, NumPy
2.5.4, and CPU backend. The UMI caller used Python 3.11.15 with NumPy 1.26.4,
pandas 2.3.3, xarray 2022.3.0, and scikit-learn 1.5.1. That existing UMI
environment reports scikit-learn outside unified's declared `>=1.7,<1.8`
compatibility range. No shared checkout, environment, cache, or worker was
changed; the new worker used `pip --no-cache-dir`.

The UMI integration and existing experiment, recording, replay, and session
tests:

```sh
MEMO_PYTHON=/private/tmp/memo-rsa-py312/bin/python \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/memo-core-pin:/private/tmp/umi-memo-demo \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest \
  -p no:cacheprovider tests/test_memo_umi.py tests/test_experiments.py \
  tests/test_record_validation.py tests/test_environment_sessions.py -q
```

The full native/UMI comparison and input-replay recording:

```sh
MEMO_PYTHON=/private/tmp/memo-rsa-py312/bin/python \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/memo-core-pin:/private/tmp/umi-memo-demo \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python \
  examples/memo_umi/run.py --out /private/tmp/memo-umi-runs-final4
```

Pinned-core subject, streaming, I/O catalog, and compatibility tests:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/memo-core-pin:/private/tmp/umi-memo-demo \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest \
  -p no:cacheprovider /private/tmp/memo-core-pin/tests/test_native_subject.py \
  /private/tmp/memo-core-pin/tests/test_streaming.py \
  /private/tmp/memo-core-pin/tests/test_io_catalog.py \
  /private/tmp/memo-core-pin/tests/test_channel_compatibility.py -q
```

Run the end-to-end example as documented in [the example README](../examples/memo_umi/README.md).

## Verification performed

- Confirmed unified base and each `install/peer-revisions.json` commit object.
- Focused integration tests plus existing UMI experiment, record-validation,
  environment-session tests: **203 passed** on Python 3.11.15 with the live
  memo worker available.
- Pinned-core subject, streaming, I/O catalog, and channel-compatibility tests:
  **71 passed** against the same core source snapshot.
- Combined verification across separate test invocations: **274 passed**. The
  suites are run separately because both repositories use a `tests` package and
  combined collection in one pytest process shadows one package with the other.
- No focused integration or pinned-core tests were skipped with the memo worker configured.
- Full precision live memo inference, exact direct/UMI equality, and the
  end-to-end recorded-input replay ran successfully on CPU. Reproduction output
  is summarized in `examples/memo_umi/results/summary.json`. Full experiment
  records and their referenced NumPy arrays are generated locally by the example
  or notebook. They are omitted from version control; the original recordings
  were preserved locally when removing the generated files from the PR.
- The selected test environment emitted a warning that its scikit-learn 1.5.1
  is outside unified's declared range. The memo/session tests do not score a
  benchmark; this warning is retained as an environment limitation.

The latest full documented offline tier (`pytest -m "unit or
integration" -q -ra --tb=short --timeout=60 -p no:cacheprovider`) passed:
**1,635 passed, 48 skipped, 63 deselected**. It used a new temporary Python 3.11
environment inheriting the existing environment read-only, with scikit-learn
1.7.2, pytest-check 3.0.3, and pytest-timeout 2.4.0 installed locally. The exact
pinned sibling sources were exported into temporary directories and supplied on
`PYTHONPATH`; memo continued to use its separate pinned worker. Network/model
access and production credentials were disabled for the test process. Local
sockets and process inspection were permitted. An initial run without those
dependencies and permissions had 30 environment-related failures; the corrected
run had none. Skips covered optional OpenPI source/client packages, maintainer
tools, protected EWoK archives, and the unavailable LanA atlas. Slow tests were
deselected. This remains offline testing, separate from facade regression
qualification.

The contract-review follow-up adds reactive-session tests for both notebook
implementations and transport regressions for partial startup/prediction replies,
blocked writes, fragmented successful replies, and invalid deadlines. A valid
utterance is answered immediately; a later invalid input fails before inference
for that input and leaves earlier responses available in the experiment record.
The scene instruction must precede utterances. The task remains restricted to
this scene and its four utterances: channel compatibility alone establishes an
interface match, not support for arbitrary tasks. Native batched calls and the
per-utterance session path retain exactly equal predictions.

The follow-up focused tests and example were run with:

```sh
MEMO_PYTHON=/private/tmp/memo-rsa-py312/bin/python \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/memo-core-pin:/private/tmp/umi-memo-demo \
/private/tmp/memo-pr-py311/bin/python -m pytest -p no:cacheprovider \
  tests/test_memo_umi.py tests/test_python_rsa_reference.py -q --timeout=20

MEMO_PYTHON=/private/tmp/memo-rsa-py312/bin/python \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/memo-core-pin:/private/tmp/umi-memo-demo \
/private/tmp/memo-pr-py311/bin/python examples/memo_umi/run.py \
  --out /private/tmp/memo-umi-contract-fixes
```

The pinned-core subject/streaming/catalog/compatibility tests were also rerun:
**71 passed**. The refreshed probability summary adds provenance and retains
exactly the same probabilities as its previous version. No contract changes
were needed.

## Scientific limits

The paper's §5 notes memo's array backend is for finite, statically known choice
domains; continuous variables require discretization and dynamic choice
sequences are unsupported. This small RSA task has finite static domains. The
integration makes no new human-data claim. The paper's cited 64% preference for
the green circle is only contextual published summary, not a fit or evaluation
set in this example.

## Remaining work

- The unified `feat/contract-v3-facade` candidate still needs its separate
  real-model regression qualification; this integration does not perform that
  qualification.
- This demonstration adds no leaderboard benchmark and makes no human-data
  comparison. A future scientific study would need a separate fitting/evaluation
  design and appropriate human observations.
