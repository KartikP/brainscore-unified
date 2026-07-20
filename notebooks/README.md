# Brain-Score UMI notebook path

The numbered notebooks are the public learning path. They use synthetic or
small in-memory data unless the table says otherwise. Install their dependencies
from the distribution root with:

~~~bash
python -m pip install -e "unified[notebooks]"
jupyter notebook unified/notebooks/
~~~

Evidence labels:

- **local workflow**: executes a real UMI path with small local components
- **illustrative companion**: teaches analysis or interpretation, not model
  registration or a brain-alignment result
- **structural demo**: verifies interface mechanics, not scientific validity
- **EC2 companion**: full data/model scoring requires EC2
- **archived**: not part of the default executable path

## Recommended order

| # | Notebook | What it shows | Evidence and hardware | Typical runtime | Prerequisites |
| --- | --- | --- | --- | --- | --- |
| 01 | 01_quickstart_layer_mapping.ipynb | Register a deterministic vision stand-in, score it, and record one/all/composite regions | local workflow, CPU | under 10 s | base notebook environment |
| 02 | 02_behavioral_and_nulls.ipynb | Compare a behavioral score with chance and random-feature floors | illustrative companion, CPU | under 10 s | matplotlib |
| 03 | 03_state_change_ablation.ipynb | Apply, observe, and exactly reset a small PyTorch ablation | local workflow, CPU | under 10 s | torch |
| 04 | 04_embodied_vlm_game.ipynb | Run a tiny neural policy in a closed loop and align per-tick activations | structural demo, CPU | about 15 s | torch, matplotlib |
| 05 | 05_temporal_multimodal.ipynb | Synchronize modality streams, convolve an HRF, and run a temporal-shift null | local workflow, CPU | under 10 s | NumPy and SciPy |
| 06 | 06_topographic_metric.ipynb | Visualize synthetic correlation-versus-distance profiles | illustrative companion, CPU | under 10 s | matplotlib |
| 07 | 07_brain_visualization.ipynb | Render an always-local parcel heatmap; optionally render a downloaded cortical surface | local default plus optional download | under 10 s locally | matplotlib; optional nilearn/network |
| 08 | 08_scaling_curves.ipynb | Plot recorded repository results against matched null floors | local results visualization, CPU | under 10 s | matplotlib |
| 09 | archive/09_watch_api_game.ipynb | Paid OpenRouter MiniGrid experiment retained for reference | archived, external API | variable and paid | API key, gymnasium, minigrid |
| 10 | 10_intervention_spectrum.ipynb | Compare global, regional, single-unit, lesion, and drive interventions | local workflow, CPU | under 10 s | torch |
| 11 | 11_multiple_subjects.ipynb | Compare several model-subjects and run the multi-agent harness | structural demo, CPU | under 10 s | torch |
| 12 | 12_bring_your_model.ipynb | Take a PyTorch model from `nn.Module` to a wired candidate: inspect, wrap, extract, scaffold | local workflow, CPU | under 30 s | torch, torchvision, Pillow |
| 13 | 13_real_model_representations.ipynb | Record a pretrained ResNet-50's V1/V2/V4/IT layers and plot how category structure sharpens along the hierarchy (RDM, MDS, separability) | local workflow, CPU | under 30 s | torch, torchvision, matplotlib, scikit-learn, Pillow |
| 14 | 14_brain_alignment.ipynb | Plot CLIP's held-out predictions of real MajajHong2015 IT neural responses (predicted-vs-measured scatter, per-site predictivity) | EC2 result, replots saved data locally | under 10 s locally | matplotlib |

Notebook 02 does not score a registered model; it is a null-interpretation
companion. Notebook 06 does not register or score a model; it demonstrates the
descriptor used by the topographic metric. Their first cells state these
boundaries explicitly.

The production counterparts for large models, public benchmark data, video,
audio-video, and Algonauts are EC2-only. Capability status is summarized in
../README.md.

## Reproduce the laptop path

Run an individual notebook without modifying the committed file:

~~~bash
cd unified/notebooks
jupyter nbconvert --to notebook --execute --stdout 01_quickstart_layer_mapping.ipynb >/tmp/01.ipynb
~~~

Maintainers should execute every active laptop notebook from top to bottom and
commit outputs only when execution succeeds with no error cells. Paid API and
EC2 notebooks do not belong in that gate.

## Archive

archive/ contains superseded, heavy, or externally gated notebooks. See
[archive/README.md](archive/README.md) before using them.
