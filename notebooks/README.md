# Notebooks

There are two kinds of notebook.

- **Lessons** teach one concept each, in order. They use tiny models and synthetic data, run on a laptop CPU in under a minute, and download nothing.
- **Experiments** use those concepts to answer one scientific question each, written as a short lab report: question, background, hypothesis, method, results, conclusion and limits.

Start with the lessons. Read the [manual](../docs/README.md) alongside them when a term is new.

## Set up

```bash
python -m pip install -e './unified[notebooks]'      # from the workspace root
jupyter notebook unified/notebooks/
```

If the result cache points to an unmounted drive, set `RESULTCACHING_HOME` to a writable folder before starting Jupyter.

## Lessons

| # | Lesson | You learn |
| --- | --- | --- |
| 1 | [Your first subject](lessons/01_your_first_subject.ipynb) | Subjects, channels, events, sessions, `participate` |
| 2 | [Your first benchmark](lessons/02_your_first_benchmark.ipynb) | Benchmarks, metrics, scores, compatibility and N/A |
| 3 | [Wrap a PyTorch model](lessons/03_wrap_a_pytorch_model.ipynb) | `BrainScoreModel`, `extractor`, `region_layer_map`; `participate` versus `process` |
| 4 | [Bring an existing Brain-Score model](lessons/04_bring_an_existing_model.ipynb) | Adapters; translating a classic registration |
| 5 | [Layers and regions](lessons/05_layers_and_regions.ipynb) | Layer paths, several regions in one pass, multi-tower routing |
| 6 | [Behavior](lessons/06_behavior.ipynb) | `TaskContext` and its `response` (`Choice`, `Scale`, `Text`); reported versus implied answers |
| 7 | [Perturb a model](lessons/07_perturb_a_model.ipynb) | `Selection`, `Perturbation`, `StateChange`, `intervention`, undo |
| 8 | [Choose units](lessons/08_choose_units.ipynb) | Functional localizers, matched random controls, `CompositeSelector` |
| 9 | [Closed loop](lessons/09_closed_loop.ipynb) | Environments, `policy`, `EnvironmentSession` |
| 10 | [Run an experiment](lessons/10_run_an_experiment.ipynb) | `Experiment`, `TaskProtocol`, recorders, conditions; one presentation for behavior and activity |

## Experiments

| # | Question | Lessons | Requires |
| --- | --- | --- | --- |
| 1 | [Does a readout beat random features?](experiments/01_does_the_model_beat_random_features.ipynb) | 2, 6 | CPU, seconds |
| 2 | [Does category structure increase along the layers mapped to V1 to IT?](experiments/02_which_layer_matches_which_region.ipynb) | 3, 5 | CPU, ResNet-50 weights (100 MB) |
| 3 | [How well do CLIP features predict macaque IT responses?](experiments/03_predicting_it_responses.ipynb) | 2, 3 | CPU to replot saved results |
| 4 | [Which image half does a digit classifier rely on?](experiments/04_silencing_units_in_a_digit_classifier.ipynb) | 7, 10 | CPU, seconds |
| 5 | [How does the effect of an ablation scale with its size and strength?](experiments/05_how_much_ablation_matters.ipynb) | 7, 8 | CPU, seconds |
| 6 | [Does a topographic metric distinguish organized from random layouts?](experiments/06_spatial_organization_of_units.ipynb) | 5 | CPU, seconds |
| 7 | [Does a model of a delayed response need correctly timed features?](experiments/07_aligning_signals_in_time.ipynb) | 1, 2 | CPU, seconds |
| 8 | [Does a policy cloned from an expert solve new games in a closed loop?](experiments/08_learning_in_a_closed_loop.ipynb) | 9 | CPU, under a minute |
| 9 | [Does silencing one residual branch change ResNet-18's predictions?](experiments/09_recording_and_silencing_resnet18.ipynb) | 7, 10 | CPU, ResNet-18 weights (45 MB) |
| 10 | [How well do GPT-2 features predict fMRI during story listening?](experiments/10_story_listening_fmri.ipynb) | 3, 5 | CPU to read saved results; rerun needs LeBel 2023 data |
| 11 | [Does silencing one internal unit change a trained robot policy's actions?](experiments/11_inside_a_robot_policy.ipynb) | 9, 10 | CPU to read saved results; rerun needs OpenPI and a GPU |
| 12 | [Is the effect of pushing one neuron in Qwen3.5-4B specific to that neuron?](experiments/12_steering_a_language_model.ipynb) | 7, 8 | GPU (16 GB), 10 GB download |
| 13 | [Do outputs passed around a loop of subjects converge?](experiments/13_passing_outputs_between_subjects.ipynb) | 1 | CPU, seconds |

Results from saved runs, synthetic data or untrained networks demonstrate a procedure. Each notebook's "Conclusion and limits" section says what its result does and does not establish.

## Folders

- `assets/` and `data/`: figures and saved results the experiments read.
- `figure_sources/`: scripts that produced the saved figures.
- `archive/`: earlier notebooks, kept unchanged for reference. They are not maintained and some of their paths no longer resolve.

## Automated check

```bash
python -m brainscore.validation.notebooks --source unified/notebooks --out /tmp/notebook-check
```

Runs every lesson and every CPU experiment in a fresh kernel without modifying the source. Notebooks that need downloads or a GPU are listed as excluded in `brainscore/validation/notebooks.py`.
