# Prepare CogGym data locally

CogGym's runner reads experiments and reference human results directly from the
authors' repository. Preparing data means obtaining that pinned checkout:
`python -m brainscore.data package coggym` clones it (see Rebuild below).
This folder documents that workflow; its `__init__.py` declares the checkout as a local
asset (`coggym-checkout`) and registers no dataset.

## 1. Obtain the reference files

After installing the four UMI packages in your experiment environment, clone
CogGym into a new directory:

```bash
# Clone the authors' repository at the pinned revision, into the folder the benchmarks read
# ($BRAINSCORE_DATA_HOME/coggym-checkout, default ~/brainscore-data/coggym-checkout)
python -m brainscore.data package coggym
# Install CogGym's own requirements
python -m pip install -r ~/brainscore-data/coggym-checkout/evaluation/requirements.txt
```

Already have the authors' files? Move that Git checkout to
`$BRAINSCORE_DATA_HOME/coggym-checkout`, or put a link there to it. It must use the revision above, with no tracked edits under `EML/`
or `evaluation/`. Use a separate checkout if you are editing CogGym itself.

The runner uses these files in their original locations:

| Path | Purpose |
| --- | --- |
| `EML/` | Experiment definitions, stimuli, and reference data |
| `evaluation/public_manifest.json` | Experiments available to the public evaluator |
| `evaluation/trial_selection_map.json` | Trials selected for each experiment |
| `evaluation/` | Prompt building, evaluation, response parsing, and scoring |

The installed requirements include the Google SDK because CogGym imports it
even for local models. A local model does not need a Google API key.

## 2. Check an experiment before loading a model

```python
import brainscore

# Validate the checkout, selected trials, and prompts. No model is called.
benchmark = brainscore.load_benchmark("CogGym.Hu2023Fine.exp1")
print(benchmark.identifier)
```

Alternatively, pass `checkout="/path/to/coggym"` to `load_benchmark()`.
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
# Load one experiment with the reset of your model, then score the model
benchmark = brainscore.load_benchmark("CogGym.Hu2023Fine.exp1", reset=lambda repetition: None)
score = benchmark(model)
```

Scores can differ a little from scores made with CogGym's own client. The model
builds its own prompt from the instruction and the parts, so a chat model can
place the instruction in the user turn, where CogGym's client sends a separate
system text. The score says so in `score.attrs['input_adaptation']`.

Supply the reset callback of a model that keeps state between repetitions.
Save run outputs outside the reference checkout. UMI records model calls while
CogGym keeps its trial selection and scoring.

There is no CogGym request-ID resolver in UMI. Access to additional datasets
does not automatically register them or extend this pinned public subset.
Follow the authors' access and distribution terms for any additional files.

## Rebuild

### coggym

- Source: CogGym (Ying et al.), https://github.com/lance-ying/coggym at revision a1cd9df1118fec80eba7463de237d1497d77e041. Licence: not stated by the repository.
- Access: public, no account.
- Needs: git; the size of the repository (not measured); minutes.
- Command: `python -m brainscore.data package coggym`. Source folder: none: the script clones the repository itself. Output (or `--output <new folder>`): the folder the benchmarks read, `$BRAINSCORE_DATA_HOME/coggym-checkout` (`~/brainscore-data/coggym-checkout` by default).
- Writes: the checkout at the pinned revision (`EML/`, `evaluation/` and the rest). Then: install `evaluation/requirements.txt` of the checkout into this environment (`pip install -r <checkout>/evaluation/requirements.txt`).
- Check: the printed counts; no sha1 is recorded for these files
