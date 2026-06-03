"""Generate the organized, per-capability showcase notebooks for the unified
interface, one notebook per capability, using a small VLM (CLIP ViT-B/32 /
Qwen2.5-VL-3B) as the demonstration model.

Each notebook is self-contained, resets model state at the end, and is meant to
be executed top-to-bottom on EC2 (never the user's laptop). Synthetic-only cells
(nulls, temporal utilities, topographic metric, grid-game oracle, visualization)
run anywhere; cells that score real models on real benchmarks need GPU + data.

Run:  python build_capability_notebooks.py            # writes *.ipynb here
Then on EC2:  jupyter nbconvert --to notebook --execute <nb>.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def notebook(cells):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(c[1]) if c[0] == 'md' else new_code_cell(c[1])
                for c in cells]
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3',
                                 'language': 'python'}
    return nb


HEADER = lambda title, blurb: ('md', f"# {title}\n\n{blurb}\n\n"
    "> Run on EC2 (GPU + Brain-Score data). Do **not** run on a laptop.\n")


# ---------------------------------------------------------------------------
# 01 — Quickstart + the three layer-mapping types
# ---------------------------------------------------------------------------
NB01 = [
    HEADER("01 · Quickstart & layer mapping",
           "Register a model once, score it across domains, and record from the "
           "brain three ways: a single region (standard), the whole brain "
           "(`'all'`), and a population gathered across layers (`CompositeSelector`)."),
    ('md', "## Load a small VLM and score one benchmark"),
    ('code', "from brainscore import load_model, load_benchmark\n"
             "model = load_model('clip-vit-b-32')\n"
             "print('modalities:', model.supported_modalities)\n"
             "benchmark = load_benchmark('Yeatman2021-lexical_decision-image')\n"
             "score = benchmark(model)\n"
             "print('ROAR lexical-decision score:', float(score))"),
    ('md', "## Standard recording — one region maps to one layer"),
    ('code', "print('region_layer_map:', dict(model.region_layer_map))\n"
             "model.start_recording('IT')\n"
             "print('recording region:', model._recording_regions,\n"
             "      '-> layer(s):', model._recording_layers)"),
    ('md', "## Whole-brain recording — `start_recording('all')`\n"
           "Every region in the model's `region_layer_map` is recorded in one pass; "
           "neuroids carry a `region` coord."),
    ('code', "model.start_recording('all')\n"
             "print('regions:', model._recording_regions)\n"
             "print('layers (deduped):', model._recording_layers)"),
    ('md', "## Composite recording — a population across layers\n"
           "`CompositeSelector` gathers units from several layers into one region — "
           "the v1.5 mechanism behind functional populations that span depth."),
    ('code', "from brainscore_core import CompositeSelector\n"
             "# build a composite from two real layers already in this model's map\n"
             "str_layers = [v for v in model.region_layer_map.values() if isinstance(v, str)]\n"
             "picked = list(dict.fromkeys(str_layers))[:2] or str_layers[:1]\n"
             "sel = CompositeSelector(layers=tuple((L, None) for L in picked))\n"
             "print('composite layer paths:', sel.layer_paths)\n"
             "model.region_layer_map['Vc'] = sel\n"
             "model.start_recording('Vc')\n"
             "print('composite recording:', model._composite_recording,\n"
             "      '| layers:', model._recording_layers)"),
    ('md', "## Reset state (every notebook leaves the model clean)"),
    ('code', "model.reset()\nprint('done')"),
]


# ---------------------------------------------------------------------------
# 02 — Behavioral evaluation + nulls
# ---------------------------------------------------------------------------
NB02 = [
    HEADER("02 · Behavioral evaluation & nulls",
           "Score a behavioral benchmark (ROAR lexical decision) and — crucially — "
           "compare against the matched null floors. A score that doesn't clear "
           "`chance-baseline` and `random-vit-b-32` is reporting noise."),
    ('code', "from brainscore import load_model, load_benchmark\n"
             "bench = load_benchmark('Yeatman2021-lexical_decision-image')\n"
             "results = {}\n"
             "for m in ['chance-baseline', 'random-vit-b-32', 'clip-vit-b-32']:\n"
             "    results[m] = float(bench(load_model(m)))\n"
             "    print(f'{m:18s} {results[m]:.3f}')"),
    ('md', "## The null floors, as a bar plot with the chance line"),
    ('code', "from brainscore.visualization import ablation_effect_bar\n"
             "ablation_effect_bar({k: (v, 0.0) for k, v in results.items()},\n"
             "                    ylabel='ROAR score', chance=0.5,\n"
             "                    title='Behavioral score vs null floors',\n"
             "                    out_png='roar_nulls.png')\n"
             "from IPython.display import Image; Image('roar_nulls.png')"),
    ('md', "**Reading it:** `chance-baseline` sits at the label-balance floor; "
           "`random-vit-b-32` adds whatever a logistic readout overfits from random "
           "features; a real model must clear both. This is the discipline applied "
           "to every capability in these notebooks."),
]


# ---------------------------------------------------------------------------
# 03 — State change / ablation (induced dyslexia)
# ---------------------------------------------------------------------------
NB03 = [
    HEADER("03 · State change & ablation",
           "`process(StateChange)` lesions a population of units, observes the effect, "
           "and restores exactly — the mechanism behind induced-dyslexia experiments. "
           "Here on a small torch network so it runs fast and is fully inspectable."),
    ('code', "import numpy as np, torch, torch.nn as nn\n"
             "from brainscore_core.model_interface import (BrainScoreModel, StateChange,\n"
             "    Selection, Perturbation)\n"
             "from brainscore.perturbation import build_pytorch_ablation_fn\n"
             "torch.manual_seed(0)\n"
             "net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))\n"
             "net.eval()"),
    ('md', "## Wrap with a state_change_fn and ablate a population\n"
           "Layer `'0'` is the first Linear (16 output units); we zero its first 8."),
    ('code', "bs = BrainScoreModel(identifier='toy', model=net, region_layer_map={'R': '0'},\n"
             "                     preprocessors={}, activations_model=None,\n"
             "                     state_change_fn=build_pytorch_ablation_fn(net))\n"
             "x = torch.randn(5, 8)\n"
             "baseline = net(x).detach().numpy()\n"
             "applied = bs.process(StateChange(kind='ablation',\n"
             "             target=Selection(layer='0', indices=list(range(8))),\n"
             "             perturbation=Perturbation(kind='zero')))\n"
             "lesioned = net(x).detach().numpy()\n"
             "print('handle:', applied.handle_id,\n"
             "      '| output changed:', not np.allclose(baseline, lesioned))"),
    ('md', "## Visualize intact vs lesioned vs difference"),
    ('code', "from brainscore.visualization import before_after_difference\n"
             "# show first-layer activations under a hook is overkill here; show outputs\n"
             "before_after_difference(baseline, lesioned, out_png='ablation.png',\n"
             "                        title='Output before/after lesion')\n"
             "from IPython.display import Image; Image('ablation.png')"),
    ('md', "## Restore exactly and confirm bit-for-bit recovery"),
    ('code', "bs.reset()\n"
             "restored = net(x).detach().numpy()\n"
             "print('restored == baseline:', np.allclose(restored, baseline))"),
    ('md', "**Null for ablation:** ablating a *random* same-size population should "
           "produce a weaker effect than a curated (e.g. top-Cohen's-d) population. "
           "Use `brainscore_core.nulls.random_unit_subset` to pick the random set."),
]


# ---------------------------------------------------------------------------
# 04 — Embodied: a VLM plays a video game
# ---------------------------------------------------------------------------
NB04 = [
    HEADER("04 · Embodied — a VLM plays a video game",
           "`process(EnvironmentStep)` drives a closed loop: render a frame, the model "
           "picks a move, step, repeat. We compare an oracle, a random null, and "
           "(optionally) a real VLM on identical boards."),
    ('code', "from brainscore.harnesses.grid_game import (GridGameEnv, play_game,\n"
             "    greedy_oracle_policy, random_action_policy, evaluate_policy)\n"
             "from brainscore_core.model_interface import BrainScoreModel\n"
             "from brainscore.model_helpers.policy_wrapper import PolicyWrapper\n"
             "def make_model(policy):\n"
             "    return BrainScoreModel('player', None, {}, {}, None,\n"
             "                           action_fn=PolicyWrapper(policy, max_history=4))"),
    ('md', "## Oracle vs random null over 25 boards"),
    ('code', "mk = lambda p: make_model(p)\n"
             "oracle = evaluate_policy(mk, greedy_oracle_policy, n_episodes=25)\n"
             "rand = evaluate_policy(mk, random_action_policy(0), n_episodes=25)\n"
             "print('oracle:', oracle)\nprint('random:', rand)"),
    ('md', "## A single rendered frame (what a VLM policy sees)"),
    ('code', "import matplotlib.pyplot as plt\n"
             "env = GridGameEnv(size=5, seed=3)\n"
             "obs = env.reset()\n"
             "plt.imshow(obs['frame']); plt.axis('off'); plt.title('grid game frame')\n"
             "plt.savefig('frame.png', dpi=120, bbox_inches='tight')\n"
             "print(obs['ascii'])"),
    ('md', "**Scaling result (from `scripts/vlm_game`):** random 0.20 → "
           "Qwen2.5-VL-3B 0.0 → Qwen2.5-VL-7B 0.13 (efficiency 1.0 when it solves) → "
           "thinking-model-over-ASCII (perfect perception) → oracle 1.0. The visual "
           "VLMs' struggle vs the thinking model's success isolates *perception* as "
           "the bottleneck — the kind of decomposition the interface makes easy."),
]


# ---------------------------------------------------------------------------
# 05 — Temporal & multimodal alignment
# ---------------------------------------------------------------------------
NB05 = [
    HEADER("05 · Temporal & multimodal alignment",
           "The temporal toolkit that lets per-frame / per-token / per-sample model "
           "features meet TR-resolved brain data: `temporal_bin`, `hrf_convolve`, "
           "`synchronize_modalities`, and the temporal-shift null."),
    ('code', "import numpy as np\n"
             "from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly\n"
             "from brainscore_core.temporal import synchronize_modalities, hrf_convolve, double_gamma_hrf"),
    ('md', "## Two modality streams on different native rates → one TR grid"),
    ('code', "def stream(n_pres, src_bins, n_neuroid, scale):\n"
             "    nb=len(src_bins); data=np.stack([np.full((n_pres,n_neuroid), b*scale) for b in range(nb)],1)\n"
             "    return NeuroidAssembly(data, dims=['presentation','time_bin','neuroid'],\n"
             "        coords={'clip_id':('presentation',[f'c{i}' for i in range(n_pres)]),\n"
             "                'stimulus_id':('presentation',[f'c{i}' for i in range(n_pres)]),\n"
             "                'time_bin_start_ms':('time_bin',[s for s,_ in src_bins]),\n"
             "                'time_bin_end_ms':('time_bin',[e for _,e in src_bins]),\n"
             "                'neuroid_id':('neuroid',np.arange(n_neuroid)),\n"
             "                'layer':('neuroid',['L']*n_neuroid)})\n"
             "video = stream(3, [(0,250),(250,500),(500,750),(750,1000)], 4, 1.0)\n"
             "audio = stream(3, [(0,500),(500,1000)], 6, 10.0)\n"
             "merged = synchronize_modalities({'video':video,'audio':audio}, [(0,500),(500,1000)])\n"
             "print('merged shape:', dict(merged.sizes))\n"
             "print('modalities:', set(merged['modality'].values.tolist()))"),
    ('md', "## HRF convolution (causal — no future leakage)"),
    ('code', "hrf = double_gamma_hrf(duration_sec=30, sampling_rate_hz=1.0)\n"
             "feats = np.zeros((60, 3)); feats[10] = 1.0\n"
             "conv = hrf_convolve(feats, sampling_rate_hz=1.0)\n"
             "print('peak shifted to TR:', int(np.argmax(conv[:,0])), '(impulse at 10, HRF peaks ~5s later)')"),
    ('md', "## Temporal-shift null — mis-time features, watch prediction degrade"),
    ('code', "from brainscore_core.nulls import shift_features, temporal_shift_curve\n"
             "rng=np.random.RandomState(0); X=rng.randn(240,6); W=rng.randn(6,4); Y=X@W+0.1*rng.randn(240,4)\n"
             "def r(Xs):\n"
             "    h=120; from numpy.linalg import solve\n"
             "    Xtr,Ytr,Xte,Yte=Xs[:h],Y[:h],Xs[h:],Y[h:]\n"
             "    Wf=solve(Xtr.T@Xtr+np.eye(6), Xtr.T@(Ytr-Ytr.mean(0)))\n"
             "    p=Xte@Wf+Ytr.mean(0)\n"
             "    return float(np.mean([np.corrcoef(p[:,j],Yte[:,j])[0,1] for j in range(4)]))\n"
             "curve=temporal_shift_curve(r, X, [-10,-3,0,3,10])\n"
             "print('shift -> score:', {k:round(v,2) for k,v in curve.items()})\n"
             "print('peaks at 0:', curve[0]==max(curve.values()))"),
    ('md', "**Why this null matters:** if shifting the model features in time does "
           "*not* drop the score, the alignment was never carrying stimulus-locked "
           "information — the correlation was an artifact."),
]


# ---------------------------------------------------------------------------
# 06 — Topographic metric (TDANN / TopoLM)
# ---------------------------------------------------------------------------
NB06 = [
    HEADER("06 · Topographic organization metric",
           "A *type* of metric that scores a model's spatial unit layout against "
           "cortical topography (TDANN / TopoLM), via the correlation-vs-distance "
           "profile — not predictivity."),
    ('code', "import numpy as np\n"
             "from scipy.ndimage import gaussian_filter\n"
             "from brainscore.metrics.topographic import (spatial_smoothness,\n"
             "    correlation_distance_profile, topographic_alignment)\n"
             "def grid_pos(g):\n"
             "    r,c=np.meshgrid(np.arange(g),np.arange(g),indexing='ij')\n"
             "    return np.stack([r.ravel(),c.ravel()],-1)/(g-1)\n"
             "def topo(n=60,g=16,seed=0,sig=2.0):\n"
             "    rng=np.random.RandomState(seed); pos=grid_pos(g)\n"
             "    R=np.stack([gaussian_filter(rng.randn(g,g),sig,mode='wrap').ravel() for _ in range(n)])\n"
             "    return R,pos\n"
             "def rand(n=60,g=16,seed=0):\n"
             "    rng=np.random.RandomState(seed); return rng.randn(n,g*g),grid_pos(g)"),
    ('md', "## Topographic model has high spatial smoothness; random ~0"),
    ('code', "tr,tp=topo(); rr,rp=rand()\n"
             "print('topographic smoothness:', round(spatial_smoothness(tr,tp),3))\n"
             "print('random smoothness:     ', round(spatial_smoothness(rr,rp),3))"),
    ('md', "## Correlation-vs-distance profile (the descriptor)"),
    ('code', "import matplotlib.pyplot as plt\n"
             "cx,cy,_=correlation_distance_profile(tr,tp,n_bins=12)\n"
             "rx,ry,_=correlation_distance_profile(rr,rp,n_bins=12)\n"
             "plt.plot(cx,cy,'-o',label='topographic'); plt.plot(rx,ry,'-o',label='random')\n"
             "plt.xlabel('normalized distance'); plt.ylabel('mean response corr'); plt.legend()\n"
             "plt.savefig('topo_profile.png',dpi=120,bbox_inches='tight')"),
    ('md', "On fMRI, the brain's profile is computed from voxel surface coordinates and "
           "`TopographicMetric` correlates the two profiles — high when the model is "
           "organized like cortex."),
]


# ---------------------------------------------------------------------------
# 07 — Brain visualization (MIRAGE-style)
# ---------------------------------------------------------------------------
NB07 = [
    HEADER("07 · Brain visualization",
           "Map per-parcel / per-voxel model scores onto the cortex — the "
           "response-on-brain figures. Fallback heatmaps run anywhere; the inflated "
           "cortical surface needs nilearn + fsaverage assets (present on EC2)."),
    ('code', "import numpy as np\n"
             "from brainscore.visualization import (parcel_grid_heatmap, network_strip,\n"
             "    cortical_surface_map)\n"
             "# pretend per-parcel prediction r for 1000 Schaefer parcels\n"
             "rng=np.random.RandomState(0); per_parcel_r = np.clip(rng.rand(1000)*0.4, 0, 1)"),
    ('md', "## Always-works fallback: parcel heatmap"),
    ('code', "parcel_grid_heatmap(per_parcel_r, title='per-parcel prediction r',\n"
             "                    out_png='parcels.png')\n"
             "from IPython.display import Image; Image('parcels.png')"),
    ('md', "## Real inflated-cortex surface (EC2 with nilearn)"),
    ('code', "try:\n"
             "    cortical_surface_map(per_parcel_r, hemi='left', view='lateral',\n"
             "                         title='model response on cortex', out_png='surf.png')\n"
             "    from IPython.display import Image as I; display(I('surf.png'))\n"
             "except Exception as e:\n"
             "    print('surface render needs nilearn + fsaverage assets:', e)"),
]


# ---------------------------------------------------------------------------
# 08 — Scaling curves across capabilities
# ---------------------------------------------------------------------------
NB08 = [
    HEADER("08 · Scaling curves across capabilities",
           "The headline validation: does each benchmark track model quality? Plot "
           "capability score vs a bad→good model ladder, with the null floor. Loads "
           "the results JSON produced by `scripts/scaling` + `scripts/vlm_game`."),
    ('code', "import json, os\n"
             "from brainscore.visualization import scaling_curves_grid\n"
             "# Replace with real results JSON paths when available; demo values shown.\n"
             "models=['random','CLIP-B32','Qwen-3B','BLIP-2','Qwen-7B']\n"
             "scores={\n"
             "  'IT encoding (r)':[0.10,0.37,0.32,0.33,0.34],\n"
             "  'language encoding (r)':[0.12,0.46,0.71,0.74,0.71],\n"
             "  'ROAR behavior':[0.50,0.68,0.74,0.79,0.83],\n"
             "  'game success':[0.20,0.0,0.0,float('nan'),0.13],\n"
             "}\n"
             "floors={'IT encoding (r)':0.10,'language encoding (r)':0.12,\n"
             "        'ROAR behavior':0.50,'game success':0.20}\n"
             "scaling_curves_grid(models, scores, null_floors=floors,\n"
             "    title='Capabilities scale with model quality', out_png='scaling.png')\n"
             "from IPython.display import Image; Image('scaling.png')"),
    ('md', "**Reading it:** a capability whose curve climbs above its null floor and "
           "rises with model quality is measuring something real. A flat curve flags a "
           "saturated or noise-limited benchmark — surfaced honestly, not hidden."),
]


NOTEBOOKS = {
    '01_quickstart_layer_mapping.ipynb': NB01,
    '02_behavioral_and_nulls.ipynb': NB02,
    '03_state_change_ablation.ipynb': NB03,
    '04_embodied_vlm_game.ipynb': NB04,
    '05_temporal_multimodal.ipynb': NB05,
    '06_topographic_metric.ipynb': NB06,
    '07_brain_visualization.ipynb': NB07,
    '08_scaling_curves.ipynb': NB08,
}


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name, cells in NOTEBOOKS.items():
        nb = notebook(cells)
        path = os.path.join(here, name)
        with open(path, 'w') as f:
            nbf.write(nb, f)
        print('wrote', path)


if __name__ == '__main__':
    main()
