# Learn the Brain-Score UMI

Start with **18** to attach tools to a pretrained model, or **01** to learn model registration and scoring. The remaining notebooks are independent examples. Read the [concepts](../docs/concepts.md) when a term is unfamiliar; the [shared vocabulary](../docs/conventions.md) applies throughout.

## Set up

Install the coordinated packages following [installation](../install/README.md), including the notebook dependencies. From a workspace containing `unified/`:

```bash
python -m pip install -e './unified[notebooks]'
jupyter notebook unified/notebooks/
```

Select that Python environment as your kernel and run cells in order. Some examples download weights; notebook 16 also needs a separately supplied dataset. Missing plotting libraries usually mean the notebook extra is not installed.

## Choose an example

### Register, record, and score

| # | Notebook | What you do | Data and requirements |
| --- | --- | --- | --- |
| 01 | [Select layers](01_quickstart_layer_mapping.ipynb) | Register, score, and record one/all/composite mappings. | Synthetic stand-in; CPU, no download. |
| 02 | [Compare with controls](02_behavioral_and_nulls.ipynb) | Interpret accuracy against random features. | Synthetic task; CPU. |
| 07 | [Connect your model](07_bring_your_model.ipynb) | Inspect, wrap, extract, and generate a registration template. | Random-weight ResNet-18; CPU. |
| 09 | [Read saved scores](09_scaling_curves.ipynb) | Compare each saved result with its own controls. | Local saved results; no new scoring. |
| 10 | [Compare with brain recordings](10_brain_alignment.ipynb) | Plot held-out predictions and measured IT responses. | Saved CLIP/MajajHong results; CPU to replot. |

### Inspect representations

| # | Notebook | What you do | Data and requirements |
| --- | --- | --- | --- |
| 03 | [Compare layers](03_real_model_representations.ipynb) | Compare response patterns across mapped regions. | Synthetic images; pretrained ResNet-50 download. |
| 04 | [Record several regions](04_multiregion_geometry.ipynb) | Collect several layers in one pass and separate by labels. | Same model/data profile as 03. |
| 05 | [Compare subjects](05_multiple_subjects.ipynb) | Compare small models and pass outputs between them. | Untrained models and synthetic inputs; CPU. |
| 06 | [Measure spatial organization](06_topographic_metric.ipynb) | Compare correlation with distance between units. | Synthetic layouts; CPU. |
| 08 | [Display a brain map](08_brain_visualization.ipynb) | Plot values assigned to cortical parcels. | Explicitly illustrative values; CPU local plots. |
| 13 | [Align signals in time](13_temporal_multimodal.ipynb) | Resample features and test timing shifts. | Synthetic streams; CPU. |
| 16 | [Predict story-listening responses](16_whole_brain_encoding.ipynb) | Score GPT-2 against held-out fMRI stories. | GPT-2, LeBel2023 pickle, sufficient memory. |

### Intervene and run sessions

| # | Notebook | What you do | Data and requirements |
| --- | --- | --- | --- |
| 11 | [Silence selected units](11_state_change_ablation.ipynb) | Use `Selection` and `StateChange` directly. | Small untrained network; CPU. |
| 12 | [Run a feedback loop](12_embodied_vlm_game.ipynb) | Train a grid policy and record its activity while it acts. | Synthetic game; CPU, not robotics qualification. |
| 14 | [Compare interventions](14_intervention_spectrum.ipynb) | Change selection size and intervention type. | Small untrained network; CPU. |
| 15 | [Control delivery](15_streaming_delivery.ipynb) | Inspect batching, windows, and clock policies. | Advanced internal-driver demonstration; CPU. |
| 17 | [Experiment with digits](17_experiment_toolbox.ipynb) | Train, record, ablate, restore, and replay. | Real bundled digits; CPU, no download. |
| 18 | [Experiment with ResNet-18](18_resnet_experiment_toolbox.ipynb) | Attach the same tools to a pretrained image model. | Real bundled photos; CPU, ~45 MB weight download. |

`Ablate` uses the same intervention implementation as notebooks 11 and 14. `RecordActivity` uses `ActivationWindow`, demonstrated directly in 05 and 12. These are composition and direct-control entry points to the same components.

For notebook 16, set `BRAINSCORE_LEBEL_PICKLE` to the supplied dataset path. A saved-result notebook does not rerun the original scoring, and a synthetic demonstration does not establish scientific validity.

## Check a notebook

Execute without overwriting its source:

```bash
cd unified/notebooks
jupyter nbconvert --to notebook --execute --stdout 18_resnet_experiment_toolbox.ipynb > /tmp/18.ipynb
```

Maintainers should retain outputs only from successful runs and state when data-dependent examples could not be executed. Follow the [example conventions](../docs/conventions.md) when adding a notebook. Superseded and externally gated examples are listed in [archive/README.md](archive/README.md).
