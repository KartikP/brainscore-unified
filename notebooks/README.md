# Brain-Score Unified — Demonstration Notebooks

Short, runnable notebooks that each demonstrate **one** feature of the unified model interface end-to-end. Use these as the entry point when learning a feature; they're meant to be readable in 5 minutes and runnable in under a minute on a laptop.

## Conventions

- **One feature per notebook.** Don't bundle. If you want to demo a second feature, write a second notebook.
- **Self-contained.** No external data files. Synthesize images, sentences, etc. inline. The reader should be able to clone the repo and run the notebook with no setup besides activating the conda env.
- **Concise.** Aim for ≤15 cells. Visualize before/after. Show the actual numbers, not just the mechanics.
- **Reset at the end.** If the notebook mutates the model, restore it. Future cells (or future readers running interactively) shouldn't inherit the side effects.
- **Short prose, dense code.** The notebook is not a tutorial — it's a worked example. Markdown cells are 1-2 sentences each, pointing at what the next code cell does and why.

## Current notebooks

| Notebook | Feature demonstrated | Time |
|---|---|---|
| `state_change.ipynb` | `process(StateChange)` — install/observe/reset a layer ablation; visualizes activation changes + behavioral effect on a CLIP caption-matching task | ~30s |

## Adding a new notebook

1. Add an entry to the table above.
2. Pick a feature that has only one notebook obligation. Examples that don't yet have one:
   - `environment_step.ipynb` — `process(EnvironmentStep)`: register a fake DROID-shaped policy, drive it through a 5-step rollout, show the action sequence.
   - `behavioral_readout.ipynb` — fit a probabilities classifier on CLIP features, show top-3 predictions on test stimuli.
   - `prefer_path.ipynb` — same model, two evaluation paths (readout vs generation), force each via `TaskContext.prefer_path`, show the score delta.
   - `temporal_binning.ipynb` — `temporal_bin` over a multi-frame stimulus, show the (clip, time_bin, neuroid) shape and how late-binning aggregates differently from early.
   - `compatibility.ipynb` — `check_compatibility` failure modes; show the error messages and how the registry rejects bad pairings.

3. Keep the notebook lean. If it grows past ~15 cells, ask whether you've drifted into "tutorial" territory and split it into a focused demo plus a separate doc.

4. Test it executes end-to-end before committing:
   ```bash
   conda activate brainscore-unified
   cd unified/notebooks
   jupyter nbconvert --to notebook --execute --inplace <name>.ipynb
   ```

## Conda env

These run in the project's `brainscore-unified` env. From the repo root:

```bash
conda activate brainscore-unified
jupyter notebook unified/notebooks/
```

Outputs are committed alongside the .ipynb so readers can preview without running.
