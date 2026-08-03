# Brain-Score UMI notebook path

The numbered notebooks are the public learning path. They use synthetic or
small in-memory data unless the table says otherwise.

The plotting and environment libraries these notebooks need are an **extra**, not part
of a plain install. From the workspace root:

~~~bash
python -m pip install -e "./unified[notebooks]"
jupyter notebook unified/notebooks/
~~~

> **`ModuleNotFoundError: No module named 'matplotlib'`** partway through a notebook means
> the extra is missing — the library installed fine, but `matplotlib`, `nilearn`,
> `gymnasium` and `minigrid` only come with `[notebooks]`. Run the command above and
> restart the kernel. (`[test]` likewise supplies `pytest` for the documented test
> commands; `"./unified[notebooks,test]"` gets both.)

**Read [`docs/concepts.md`](../docs/concepts.md) first** — about ten minutes, and it
defines every term these notebooks use (subject, assembly, neuroid, region_layer_map,
ceiled vs. raw). Skipping it is the main reason people stall around notebook 05.

## Four arcs

The numbering follows the reading order: each arc is a contiguous block and the numbers
run forward. You do not have to do all fifteen — after Arc 1, jump to whichever arc
matches your goal.

**Arc 1 · The core loop (01–02) — start here, everyone.**
`01` is the loop itself: register a model, record a region, process stimuli, read a
score. `02` is a short companion on *interpreting* a score against chance and
random-feature floors; it plots recorded numbers and calls no Brain-Score API, so read it
for the idea rather than the mechanics. Everything later assumes `01`.

**Arc 2 · Your own analysis (03–06) — "I want to answer my own question."**
Record real layers and look at the representations directly. `03` introduces
representational dissimilarity matrices on a pretrained ResNet-50, `04` records two
regions in one forward pass and compares their geometry, `05` compares several subjects,
`06` covers topography. None of these produce a `Score` — this is Brain-Score used as a
toolbox.

**Arc 3 · Bring your own model (07–10) — "I want to score my model."**
Wrap an `nn.Module` and wire it up (`07`), see where a model predicts the brain (`08`),
read results against their null floors (`09`), and inspect real held-out predictions
(`10`). Pair with [`../EXTENDING.md`](../EXTENDING.md) and the runnable skeletons in
[`../templates/`](../templates/).

**Arc 4 · Beyond static images (11–15) — "my work isn't feedforward vision."**
Lesions and perturbations (`11`, `14`), closed-loop embodied agents (`12`), temporal and
multimodal alignment (`13`), streaming delivery (`15`). These are independent of each
other; take only what you need.

Evidence labels:

- **local workflow**: executes a real UMI path with small local components
- **illustrative companion**: teaches analysis or interpretation, not model
  registration or a brain-alignment result
- **structural demo**: verifies interface mechanics, not scientific validity
- **EC2 companion**: full data/model scoring requires EC2
- **archived**: not part of the default executable path

## Full manifest

Numeric order below; the arcs above say which ones you actually need.

| # | Notebook | What it shows | Evidence and hardware | Typical runtime | Prerequisites |
| --- | --- | --- | --- | --- | --- |
| 01 | 01_quickstart_layer_mapping.ipynb | Register a deterministic vision stand-in, score it, and record one/all/composite regions | local workflow, CPU | under 10 s | base notebook environment |
| 02 | 02_behavioral_and_nulls.ipynb | Compare a behavioral score with chance and random-feature floors | illustrative companion, CPU | under 10 s | matplotlib |
| 03 | 03_real_model_representations.ipynb | Record a pretrained ResNet-50's V1/V2/V4/IT layers and plot how category structure sharpens along the hierarchy (RDM, MDS, separability) | local workflow, CPU | under 30 s | torch, torchvision, matplotlib, scikit-learn, Pillow |
| 04 | 04_multiregion_geometry.ipynb | Record V4 and IT in one forward pass, split the result by region, and compare their representational geometry — a whole analysis with no `Score` object | local workflow, CPU | under 60 s | torch, torchvision, matplotlib, scikit-learn, pandas, Pillow |
| 05 | 05_multiple_subjects.ipynb | Compare several model-subjects and run the multi-agent harness | structural demo, CPU | under 10 s | torch |
| 06 | 06_topographic_metric.ipynb | Visualize synthetic correlation-versus-distance profiles | illustrative companion, CPU | under 10 s | matplotlib |
| 07 | 07_bring_your_model.ipynb | Take a PyTorch model from `nn.Module` to a wired candidate: inspect, wrap, extract, scaffold | local workflow, CPU | under 30 s | torch, torchvision, Pillow |
| 08 | 08_brain_visualization.ipynb | Render an always-local parcel heatmap; optionally render a downloaded cortical surface | local default plus optional download | under 10 s locally | matplotlib; optional nilearn/network |
| 09 | 09_scaling_curves.ipynb | Plot recorded repository results against matched null floors | local results visualization, CPU | under 10 s | matplotlib |
| 10 | 10_brain_alignment.ipynb | Plot CLIP's held-out predictions of real MajajHong2015 IT neural responses (predicted-vs-measured scatter, per-site predictivity) | EC2 result, replots saved data locally | under 10 s locally | matplotlib |
| 11 | 11_state_change_ablation.ipynb | Apply, observe, and exactly reset a small PyTorch ablation | local workflow, CPU | under 10 s | torch |
| 12 | 12_embodied_vlm_game.ipynb | Run a tiny neural policy in a closed loop and align per-tick activations | structural demo, CPU | about 15 s | torch, matplotlib |
| 13 | 13_temporal_multimodal.ipynb | Synchronize modality streams, convolve an HRF, and run a temporal-shift null | local workflow, CPU | under 10 s | NumPy and SciPy |
| 14 | 14_intervention_spectrum.ipynb | Compare global, regional, single-unit, lesion, and drive interventions | local workflow, CPU | under 10 s | torch |
| 15 | 15_streaming_delivery.ipynb | Separate streaming *shape* from *delivery*: batched vs one-at-a-time (identical values), windowed delivery with memory bounded independently of feed length, and the three real-time policies | structural demo, CPU | under 10 s | none beyond base |

Notebook 02 does not score a registered model; it is a null-interpretation
companion. Notebook 06 does not register or score a model; it demonstrates the
descriptor used by the topographic metric. Their first cells state these
boundaries explicitly.

Notebook 15 uses deterministic stand-in subjects and filename strings for "frames";
it demonstrates interface mechanics, not a scientific result. The same machinery
pointed at a decoded video with real weights needs a GPU host and is not part of the
laptop path.

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
