# Performance qualification

## Local instruction-model check

The registered `qwen2.5-0.5b-instruct` checkpoint at revision
`7ae557604adf67be50417f59c2c2f167def9a775` was loaded locally on MPS in float16.
Four user-provided text inputs produced raw responses through `response_trace`.
The first run took 2.73 seconds; a subsequent warm run took 0.67 seconds. These
are experiment runtimes, not overhead estimates. Neural recording from the same
loaded subject returned a (4, 896) assembly.

Seven alternating matched warm pairs subsequently measured 450.94 ms direct
versus 437.99 ms through the GUI, or -2.87% added runtime. This is timing
variation, not evidence of a speedup. Encoded outputs matched exactly. Both
paths used the same checkpoint, device, dtype, recording, and generation
settings; startup and browser rendering were excluded. The local raw report is
`qwen-performance-result.json`. The 25% runtime gate passed for this workload;
this is not a general bound on GPU memory or other experiments.

The responses were `No`, `Yes.`, `Yes.`, and `no` for the Earth-life question,
cat/mammal question, fish/mammal question, and "Say no" respectively. The Earth
and fish answers were incorrect. A direct Transformers run on CPU in float32
with the same prompts returned the same strings. The GUI preserves these
outputs rather than treating successful execution as factual correctness.

The app suite now has 35 passing tests. It covers the exact four-input null
case (all ties), unique/tied/nonfinite preferences, restored run settings through
browser inspection, and plain-text generation preserving raw responses. The
updated unified import test verifies the new registration remains lazy. The
broader unified unit suite again reported 1472 passed and the same unrelated
workspace certificate-hygiene failure.

## Registry-backed workbench

The expanded app was rechecked on 2026-11-02 using the app-local qualified core
at `bdda009` and scikit-learn 1.7.2. Nine alternating pairs and ten repetitions
per condition gave:

| Circuit / recording | Direct median | GUI median | Added runtime |
| --- | ---: | ---: | ---: |
| Behavior | 85.61 ms | 86.30 ms | 0.82% |
| Activity | 135.15 ms | 128.35 ms | -5.03% |

Both gates passed and encoded outputs matched exactly. Negative values are
timing variation, not a claimed speedup. The raw local report is
`workbench-performance-result.json`. Catalog discovery happens before warm runs;
the catalog is fetched separately and is absent from status polling. This still
qualifies the small circuit's execution path, not every registered model,
large-model GPU inference, or browser memory.

The expanded suite has 32 passing tests, including direct-score parity through
the existing scorer, registered-model loading, failed-load recovery, neural and
behavioral stimuli, request channels, a newly registered JSON family, partial
matches, N/A handling, uploaded assets, and full-output restoration. TypeScript
builds and type-checks. Chrome was used to load `chance-baseline` and execute the
two user-entered text stimuli, returning actual 0.5/0.5 BehavioralAssembly rows.
Headless execution of that exported plan also passed. A binary compatibility
warning appears during the transformer fixture test; its cause
has not been established.

The broader unified unit suite, run in this isolated runtime with loopback and
process inspection enabled, finished with 1472 passed, 39 skipped, and one
failure. The failure is the workspace-root hygiene check matching certificate
bundles in pre-existing environments under `work/`; no finding was under the
GUI's paths. Those unrelated environments were left untouched.

## Earlier prototype measurement

Measured 2026-11-02 on macOS 27.0.1 arm64, Python 3.11.15, PyTorch 2.13.0,
transformers 4.57.6. These are local CPU measurements on small fixtures, not
pretrained large-model or GPU qualification.

| Profile / recording | Direct median | GUI median | Added runtime |
| --- | ---: | ---: | ---: |
| Eight-unit circuit / behavior | 178.10 ms | 190.95 ms | 7.22% |
| Eight-unit circuit / activity | 250.86 ms | 249.88 ms | -0.39% |
| Local random GPT-2 architecture / behavior | 155.64 ms | 148.53 ms | -4.57% |
| Local random GPT-2 architecture / activity | 175.57 ms | 170.23 ms | -3.04% |

Negative values reflect measurement variation; they are not evidence of a GUI
speedup. The gate passed in all four configurations. Encoded protocol outputs
and report values were identical across all measured runs. Direct-versus-GUI
event outputs are also compared exactly in the tests.

The circuit measurement used 11 alternating pairs and 20 repetitions per
condition. The transformer measurement used seven alternating pairs and three
repetitions per condition. Its checkpoint had two layers, two heads, a 16-wide
hidden state, a 25-token vocabulary, and deterministic random initialization.
It exercised the real checkpoint loader, causal-LM builder, forward calls,
activation hooks, interventions, response traces, and disk recording.

Both paths used the same persistent model worker, stimuli, tools, inference
settings, and artifacts. The GUI path additionally published progress metadata
and served HTTP state requests every 50 ms, ten times the normal active-tab rate.
Timing includes experiment construction, validation, resets, evaluation, result
encoding, recording, checksums, and manifest writes. It excludes initial model
loading, post-run summary generation, browser rendering, and explicit artifact
downloads. It measures experiment runtime, not the browser's polling delay before
showing a completed result.

The raw samples are in the local ignored `performance-result.json` and
`transformer-performance-result.json` files. Re-run the budget check with the
intended profile and device before treating it as qualified. The benchmark
returns a failure exit code if median added runtime exceeds 25% or encoded
outputs differ. The initial measurements do not establish a 25% bound on browser
or GPU memory, nor on the scientific cost of enabling recording itself.

At that stage, 16 app tests and TypeScript build/type check passed. Chrome was used to attach
tools by button, execute all four conditions, inspect result cards, and verify
configuration restoration after reload. Native drag automation was interrupted,
so drag-and-drop is implemented but not verified by the browser automation run.

The existing unified unit suite reached 87 passed, two skipped, and five failures
before its configured early stop. Four failures were the pre-existing mismatch
between `SessionProtocol`'s required `complete` receipt and the checked-out core's
old buffered session (that class is now the `Session` that `stimulus_session` returns); another was an unavailable result-cache directory. Focused
existing experiment tests reproduce the same completion mismatch. This app adds
its own finite-session completion adapters and does not edit library code.
The environment also reports an existing scikit-learn 1.5.2 pin outside UMI's
declared bounds; these profiles do not use scikit-learn.
