# CogGym: its own folder and environment

CogGym's evaluator (its trial selection, prompts, answer parsing and scoring) runs
in its own Python environment, in a separate process. Each model call comes back to
your process as one request, and your model answers it there: the model, the
recordings and the tools stay in your process. Nothing is installed into the
environment you score in, so its environment fingerprint does not change. This works on
Linux and macOS (it needs POSIX process groups and pipes), not on Windows.

## 1. The folder Brain-Score makes

On a fresh machine, the first `load_benchmark("CogGym...")` makes one folder:

- `checkout/`: the authors' repository (`https://github.com/lance-ying/coggym`) at
  commit `a1cd9df1118fec80eba7463de237d1497d77e041`;
- `env/`: a virtual environment with only CogGym's packages, each at an exact
  version (`brainscore/harnesses/coggym_requirements.txt`, installed with
  `--no-deps` and checked with `pip check`). The list is CogGym's own requirements
  (google-genai, python-dotenv, Pillow) and their dependencies at the versions of the
  earlier CogGym runs; `pip check` at creation is the test that it is complete on
  your platform;
- `ready.json`: the commit, the requirements' hash, the Python version and the
  environment's package list.

The folder is `~/.brain-score/environments/CogGym/<commit>-<hash>` (the hash covers the
requirements and the Python version); set `BRAINSCORE_ENVIRONMENTS_HOME` to an absolute
path to put the folders of every outside library somewhere else, or pass `folder=...` to
`load_benchmark` to name one. The general mechanism is described in
`brainscore/harnesses/README.md`.

How the folder is made:
- in a temporary folder next to it (`.<name>.partial-*`), published in one step only
  when it is complete; a folder that exists, or appears meanwhile, is never replaced
  or changed;
- git and pip get only the variables they need (user, locale, temporary folder,
  certificates, proxies, package index); no pip configuration file is read, and pip
  refuses to run outside the folder's environment, so nothing is installed elsewhere;
- a failed step (clone, environment, install, check) removes its temporary folder and
  gives one error with the step's output;
- if the process that makes the folder is killed, its git or pip command is stopped,
  and its temporary folder stays until the next creation removes it;
- a lock file next to the folder (`.<name>.lock`, kept) lets one creation run at a time.

To switch this off, set `BS_INSTALL_DEPENDENCIES=no`. A CogGym load with no ready
folder then stops before any compute, with one error that names the benchmark, the
folder and what to run. The setting covers only this folder: Brain-Score never
installs into the scoring environment.

Every score carries the record of CogGym's environment (commit, requirements hash,
hash of its package list) in `configuration['evaluator_environment']`; the
environment fingerprint of the score covers the scoring environment only.

CogGym's evaluation code imports the Google SDK even for a local model. A local model
does not need a Google API key.

## 2. Check an experiment before loading a model

```python
import brainscore

# Make or find CogGym's folder, then check the selected trials and prompts. No model is called.
benchmark = brainscore.load_benchmark("CogGym.Hu2023Fine.exp1")
print(benchmark.identifier)
```

The integration registers 23 experiments from this pinned public subset:
11 text, 6 image, and 6 video experiments. The selected experiment's media must
be present; prompt preparation checks it before inference.

## 3. Run with your model

Any model that takes the experiment's parts and writes an answer runs it: a
`BrainScoreModel` with a `reply` part (from `chat_parts`, `build_causal_lm_subject`
or `build_api_reply`; the registered closed API models) and the
inputs the trials show. Each CogGym call reaches the model on the common
channels: the system text is the task's instruction, the message parts are one
stimulus row (`text`, `vision:image`, `vision:video`) in CogGym's order, and the
temperature and length are settings on the task's `Text` response. The model
answers on `behavior:text`, and CogGym parses the answer. A subject that lacks a
media form runs with that form dropped. The score records `match: partial` and
the dropped inputs in `score.attrs['compatibility']`. Such a score is a control
for the dropped input, not a result for the full experiment. A subject that
takes none of the experiment's inputs, or cannot answer on `behavior:text`, is
N/A before any compute.

```python
benchmark = brainscore.load_benchmark("CogGym.Hu2023Fine.exp1", reset=lambda repetition: None)
score = benchmark(model)
```

Scores can differ a little from scores made with CogGym's own client. The model
builds its own prompt from the instruction and the parts, so a chat model can
place the instruction in the user turn, where CogGym's client sends a separate
system text. The score says so in `score.attrs['input_adaptation']`.

Supply the reset callback of a model that keeps state between repetitions.
Save run outputs outside CogGym's folder. UMI records model calls while
CogGym keeps its trial selection and scoring. Brain-Score gives each call the id
of its trial by a wrap of CogGym's `build_messages` that reads the trial and
changes nothing; this holds for the pinned commit, and a change of the pin must
check it again with the real CogGym.

There is no CogGym request-ID resolver in UMI. Access to additional datasets
does not automatically register them or extend this pinned public subset.
Follow the authors' access and distribution terms for any additional files.

## Rebuild

### coggym

- Source: CogGym (Ying et al.), https://github.com/lance-ying/coggym at a pinned commit. The repository holds `LICENSE` (MIT) and `DATA_LICENSE.md`.
- Access: public, no account.
- Needs: git and the packages of `brainscore/harnesses/coggym_requirements.txt`, in a folder of its own.
- Command: none: nothing to package.
- Writes: Nothing to package: the first `load_benchmark("CogGym...")` makes the checkout and its environment (see this plugin's README).
- Check: nothing to compare.
