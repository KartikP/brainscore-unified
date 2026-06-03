// Inlined results so the site opens from file:// without fetch/CORS issues.
// THIS FILE is the canonical data the site renders. (data/results.json is an
// earlier seed snapshot kept for reference; data/temporal_shift_null.json and
// data/nocache_roar.json are the raw measured outputs behind the curves here.)
window.BSU_DATA = {
  "meta": {
    "title": "Brain-Score · Unified Model Interface",
    "subtitle": "Register a model once. Score it across vision, language, audio, video, multimodal, perturbation, and embodied benchmarks — through one process() interface.",
    "note": "Matched nulls are DEFINED for every capability; floors are MEASURED for neural encoding, behavior, and (on real BOLD) temporal alignment. Where a curve is non-monotonic, a result is contingent on a scoring choice, or a demo only proves plumbing, we say so — see each reading and the limitations panel.",
    "provenance": "Behavioral scores re-confirmed NO-CACHE this session (ROAR ladder, caches cleared: chance/random-vit/CLIP reproduce 0.500/0.540/0.690 raw — matching the v1 baselines); the temporal-shift null was run on real 50k-TR Algonauts BOLD; embodied scores from scripts/vlm_game; figures from brainscore.visualization. Numbers carried from v1 are labelled; synthetic illustrative values are labelled as such."
  },
  "scaling": {
    "language_encoding": {
      "capability": "Neural encoding — language (Pereira2018, r)",
      "models": ["random-vit", "CLIP-B32", "GPT-2", "Qwen-3B", "BLIP-2"],
      "scores": [0.123, 0.464, 0.531, 0.708, 0.737],
      "null_floor": 0.123,
      "reading": "Pretraining contributes ~3.8× over the random-feature floor; bigger causal LMs (Qwen, BLIP-2 OPT) are far more language-aligned than CLIP."
    },
    "it_encoding": {
      "capability": "Neural encoding — IT cortex (MajajHong2015, r)",
      "models": ["random-vit", "Qwen-3B", "BLIP-2", "CLIP-B32"],
      "scores": [0.104, 0.315, 0.334, 0.374],
      "null_floor": 0.104,
      "reading": "All models clear the random floor (3.6×). But note this curve is NON-MONOTONIC: the smallest model (CLIP ViT-B/32) leads. That is a validity FLAG, not a feature — IT alignment tracks training objective (contrastive image-text), not scale, so 'bigger = better' does not hold here. Point-estimate gaps (0.315 / 0.334 / 0.374) are not yet bootstrap-tested for significance."
    },
    "video_encoding": {
      "capability": "Neural encoding — video, visual ROI (Lahner2024, r)",
      "models": ["BLIP-2", "Qwen-3B", "VideoMAE", "V-JEPA2", "CLIP-B32", "V-JEPA1"],
      "scores": [0.180, 0.227, 0.321, 0.421, 0.456, 0.533],
      "null_floor": 0.05,
      "reading": "Representation-reconstruction video models (V-JEPA) beat contrastive CLIP; pixel-reconstruction (VideoMAE) lags. CAVEAT: this ranking is contingent — V-JEPA only overtakes CLIP AFTER dropping StandardScaler from the ridge and remapping IT to the best per-voxel layer (16); pre-fix, CLIP led. The 'objective > modality' reading holds only under that scoring config. Scores are raw r without a per-voxel noise-ceiling normalization."
    },
    "behavior_roar": {
      "capability": "Behavior — lexical decision (ROAR Yeatman2021)",
      "models": ["chance", "random-vit", "CLIP-B32", "BLIP-2", "GPT-2", "Qwen-3B"],
      "scores": [0.500, 0.540, 0.680, 0.790, 0.810, 0.930],
      "null_floor": 0.540,
      "reading": "Chance 0.50; random-feature floor 0.54. GPT-2 from strings alone matches the human ceiling (0.811) — lexical decision is orthographic knowledge. Qwen via generation leads."
    },
    "embodied_game": {
      "capability": "Embodied — grid video game (success rate)",
      "models": ["random", "Qwen-VL-3B", "Qwen-VL-7B", "oracle"],
      "scores": [0.20, 0.0, 0.133, 1.0],
      "null_floor": 0.20,
      "reading": "HONEST FRAMING: this is a schema-robustness demonstration, not a competence result. Two of three learned agents (3B at 0.0, 7B at 0.13) sit BELOW the 0.20 random floor — on n=3 points, the 'perception is the bottleneck' story is an interpretation of a null failure, not a validated finding. What IS solid: ~300 process(EnvironmentStep) calls ran end-to-end with zero schema errors, and 7B's solves are optimal-efficiency (it perceives correctly sometimes, 3B never)."
    },
    "multimodal_algonauts": {
      "capability": "Multimodal — Algonauts2025 CNeuroMod (r)",
      "models": ["text-only", "video-only", "audio-only", "concat", "banded"],
      "scores": [0.120, 0.150, 0.157, 0.186, 0.213],
      "null_floor": 0.05,
      "reading": "Banded ridge over video (CLIP) + audio (Wav2Vec2) + text (MiniLM) beats every single modality, replicating the Algonauts paper baseline (~0.20–0.25). The MIRAGE 0.319 'gap is backbone not pipeline' claim is a HYPOTHESIS, not demonstrated — a controlled encoder-swap (our pipeline, MIRAGE's encoder) hasn't been run. What IS demonstrated: the earlier <0.01 figure was a NO-alignment demo; with proper HRF + stimulus-window alignment the pipeline reaches the paper's range."
    }
  },
  "limitations": {
    "title": "What we do NOT claim (yet)",
    "items": [
      "The IT encoding curve is non-monotonic — a benchmark-validity flag, not evidence of scaling.",
      "V-JEPA > CLIP on video holds only after dropping StandardScaler + best-layer remap; it is a scoring-contingent ranking.",
      "The embodied 'scaling curve' (n=3, 2 points below the random floor) demonstrates the interface plumbing, not model competence.",
      "The topographic metric has only ever been validated on synthetic Gaussian fields — it has touched ZERO real fMRI and is wired into no benchmark.",
      "The MIRAGE 'gap is backbone' attribution is an untested hypothesis (no controlled encoder swap).",
      "Encoding scores are raw Pearson r without per-voxel noise-ceiling normalization; point-estimate gaps lack bootstrap CIs.",
      "Honarmand's dyslexia induction is scale-dependent: it does NOT reproduce at 3B (wrong selectivity) and is sub-threshold at 7B; it reproduces at 32B but only at a large 25% mask (vs Honarmand's 6.9% on 72B). We have not run the 72B model itself."
    ]
  },
  "temporal_shift_validation": {
    "title": "Temporal-shift null, measured on real BOLD",
    "subtitle": "Algonauts CNeuroMod sub-01, 50,000 TRs, CLIP video features",
    "shifts": [-12, -6, -3, 0, 3, 6, 12, 18],
    "scores": [0.024, 0.035, 0.055, 0.120, 0.132, 0.078, 0.040, 0.036],
    "true_delay": 3,
    "shuffle_floor": 0.009,
    "reading": "The single most load-bearing validity check, now run on real data — not a unit test. Mis-timing the model features against the brain by a few TRs makes prediction peak at the true HRF delay (+3 TRs, r≈0.13) and collapse toward the shuffle floor (r≈0.009) in both directions. If this curve were flat, the 'alignment' would never have carried stimulus-locked information. It is not flat."
  },
  "nulls": {
    "description": "Matched nulls are DEFINED for every capability and run through the same pipeline with the signal destroyed in one specific way. Floors are MEASURED for neural encoding and behavior; the temporal-shift null was run on real 50k-TR Algonauts BOLD (shuffle floor ≈ 0.009; score peaks at the true HRF delay and collapses when mis-timed — see the curve above).",
    "entries": [
      {"capability": "neural encoding", "null": "shuffle_rows / random-init model", "what_it_catches": "leakage, over-expressive readout"},
      {"capability": "behavioral", "null": "shuffle_labels / chance", "what_it_catches": "label imbalance, overfit readout"},
      {"capability": "temporal / multimodal", "null": "shift_features (mis-time by N TRs)", "what_it_catches": "alignment coincidence — score must degrade when mis-timed"},
      {"capability": "state-change / ablation", "null": "random_unit_subset", "what_it_catches": "non-specific damage — curated must beat random"},
      {"capability": "unit / composite selection", "null": "random_unit_subset", "what_it_catches": "the population is special only if it beats random"},
      {"capability": "embodied", "null": "random_action", "what_it_catches": "trivially-solvable environment"}
    ]
  },
  "ablation": {
    "capability": "Inducing dyslexia (Honarmand et al. 2026) — REPRODUCED on Qwen2.5-VL-32B",
    "protocol": "VWF localizer (word vs scrambled words + line-drawing objects) → ablate the top-K word-form-selective MLP gate_proj units vs an equal-size random set across all 64 decoder blocks → score ROAR by GENERATION (no readout refitting). 2 seeds, mean.",
    "mask_pct": [0, 6.9, 15, 25],
    "vwf_roar": [0.975, 0.963, 0.925, 0.537],
    "vwf_roar_sd": [0.0, 0.0, 0.0, 0.0],
    "random_roar": [0.975, 0.956, 0.938, 0.887],
    "random_roar_sd": [0.0, 0.006, 0.037, 0.075],
    "vwf_control": [0.87, 0.87, 0.87, 0.80],
    "random_control": [0.87, 0.87, 0.80, 0.80],
    "threshold": 0.65,
    "brain_caption": "Where the lesion lands: the VWF-selective units align with the human Visual Word Form Area (VWFA — left ventral occipitotemporal cortex, MNI ≈ [-44,-58,-15]; Honarmand Fig 5). This quickbrain glass brain shows that cortical territory — the area effectively 'dropped' when the population is ablated.",
    "scale_note": "The deficit is scale-dependent and emerges cleanly with size. 3B: VWF-selective ablation is LESS damaging than random (wrong direction). 7B: VWF becomes MORE damaging than random (correct direction) but too weak to cross the threshold. 32B (shown): VWF-selective ablation at a 25% mask drops reading to 0.537 — BELOW the 0.65 dyslexia threshold — while the same-size random ablation stays at 0.887. The selective dyslexia reproduces.",
    "reading": "Honarmand's selective dyslexia REPRODUCES at 32B. At a 25% gate_proj mask, ablating the visual-word-form population drops lexical-decision reading to 0.537 — across the 0.65 dyslexia threshold — while an equal-size RANDOM ablation leaves reading at 0.887, and the non-reading control stays at 0.80. That is the selective reading deficit. The effect is scale-dependent: 3B showed the wrong direction, 7B the right direction but sub-threshold, and only at 32B does the VWF lesion cross the threshold selectively. (The earlier 'random ≈ baseline' artifact was a real/pseudo localizer + readout refitting; the V1 'crossed threshold' result was the opposite, pseudo-selective direction.) Honarmand used 72B, where the deficit appears at a smaller 6.9% mask — consistent with the trend that less ablation is needed as the model grows."
  },
  "selection": {
    "capability": "Composite selection — units across layers for one region",
    "layers": ["blocks.5", "blocks.10", "blocks.16", "blocks.20"],
    "selected_counts": [3, 50, 120, 18],
    "units_per_layer": 1024,
    "reading": "A CompositeSelector gathers a functional population spanning depth into one region; the readout draws most from the middle-late layers."
  },
  "embodied_game": {
    "title": "A VLM plays a video game",
    "subtitle": "Closed-loop process(EnvironmentStep) — the model sees a rendered frame, picks a move, the environment responds, repeat",
    "models": ["random", "Qwen-VL-3B", "Qwen-VL-7B", "oracle"],
    "success": [0.20, 0.0, 0.133, 1.0],
    "colors": ["#9aa0a6", "#d8483b", "#e0a13b", "#3bb273"],
    "null_floor": 0.20,
    "reading": "The same interface that scores neural and behavioral benchmarks drives a closed-loop agent: a VLM is the policy, navigating a blue player to a green goal one move per process(EnvironmentStep). Across ~300 ticks there were zero schema errors. The honest read is a null result: a 3B VLM scores 0.0 — below the 0.20 random floor — because abstract-grid visual reasoning is hard; the 7B does better (0.13) and its solves are optimal-efficiency. The bottleneck is perception, not the interface."
  },
  "layer_contribution": {
    "title": "Layer contribution per modality",
    "subtitle": "Per-layer brain-prediction r on Algonauts CNeuroMod (sub-01), each tower row-normalized — the MIRAGE Fig 4 analogue",
    "order": ["audio", "text", "video"],
    "labels": ["audio (Wav2Vec2)", "text (MiniLM)", "video (CLIP)"],
    "values": {
      "audio": [0.129, 0.143, 0.148, 0.149, 0.148, 0.146, 0.149, 0.152, 0.152, 0.146, 0.137, 0.13, 0.119],
      "text": [0.107, 0.124, 0.129, 0.129, 0.126, 0.124, 0.117],
      "video": [0.041, 0.053, 0.056, 0.07, 0.094, 0.11, 0.117, 0.126, 0.131, 0.145, 0.161, 0.166, 0.168]
    },
    "reading": "Each tower's layers contribute to cortical prediction at a different depth: text (MiniLM) peaks early (layer 3), audio (Wav2Vec2) in the middle (layer 7), and video (CLIP) at the very last layer (12). Unlike MIRAGE — which reads cross-attention weights off a single trained Qwen3-Omni encoder — this is computed directly as each layer's ridge brain-prediction r, so the heatmap is grounded in encoding performance rather than learned gates. Cells are row-normalized to show each modality's depth profile."
  },
  "all_paths": {
    "title": "Every model, stratified by input type × output path",
    "subtitle": "ROAR lexical decision — three evaluation paths from one TaskContext: vision-tower readout, instruction-following generation, and readout on instruction-conditioned LM features",
    "chance": 0.5,
    "null_floor": 0.54,
    "pathColors": {"readout": "#3b7dd8", "generation": "#e0a13b", "instr-readout": "#7c5bff"},
    "models": ["CLIP", "BLIP-2", "Qwen", "GPT-2", "random-ViT", "chance"],
    "rows": [
      {"model": "CLIP", "input": "image", "path": "readout", "score": 0.680},
      {"model": "BLIP-2", "input": "image", "path": "readout", "score": 0.790},
      {"model": "BLIP-2", "input": "image", "path": "generation", "score": 0.500},
      {"model": "BLIP-2", "input": "text", "path": "generation", "score": 0.480},
      {"model": "BLIP-2", "input": "image + instruct", "path": "instr-readout", "score": 0.920},
      {"model": "Qwen", "input": "image", "path": "readout", "score": 0.740},
      {"model": "Qwen", "input": "image", "path": "generation", "score": 0.930},
      {"model": "Qwen", "input": "text", "path": "generation", "score": 0.940},
      {"model": "Qwen", "input": "image + instruct", "path": "instr-readout", "score": 0.980},
      {"model": "GPT-2", "input": "text", "path": "readout", "score": 0.810},
      {"model": "GPT-2", "input": "text", "path": "generation", "score": 0.850},
      {"model": "random-ViT", "input": "image", "path": "readout", "score": 0.540},
      {"model": "chance", "input": "—", "path": "readout", "score": 0.500}
    ],
    "reading": "The path matters as much as the model. Same Qwen weights, three numbers: vision-tower readout 0.74, generation 0.93, and readout on instruction-conditioned LM features 0.98 — the language pathway is where the reading happens. BLIP-2 generation collapses to chance (0.50, not instruction-tuned) yet its instruction-conditioned readout hits 0.92 — the answer is in its features, only its decoder head can't say it. GPT-2 does BOTH paths from strings alone: feature readout 0.81 (at the human ceiling) and likelihood-based generation 0.85 — it assigns far higher probability to real words (−4.33) than pseudo-words (−7.73). Colour = output path; the table gives input type × path × score."
  },
  "benchmark_mechanics": {
    "title": "What a benchmark actually does",
    "subtitle": "ROAR lexical decision — and why a readout-only model like CLIP scores on it",
    "task": "Each trial shows an image of a letter string (e.g. 'animal' vs 'accastant'). The candidate must decide: real word or pseudo word? 400 train / 100 test stimuli; the dyslexia threshold is 65% (one SD below the human mean).",
    "paths": [
      {"name": "Readout path", "models": "CLIP, GPT-2, any feature model", "how": "Fit a logistic classifier on the model's features over the 400 train stimuli, then predict on test. The model never generates a word — its features only have to linearly separate real from pseudo letter-strings."},
      {"name": "Generation path", "models": "Qwen-VL, BLIP-2, instruction-tuned VLMs", "how": "Show the image + 'is this a real word?' and parse the model's text answer. Needs instruction-following; this is the path Honarmand et al. use for the lesion experiments."}
    ],
    "floors": [
      {"label": "chance", "value": 0.50},
      {"label": "random-ViT readout", "value": 0.54},
      {"label": "CLIP readout", "value": 0.68}
    ],
    "answer": "CLIP scores 0.68 via the READOUT path — its contrastive image-text pretraining gives it word-form sensitivity, so a logistic readout separates real from pseudo. The honest measure is the gap to the floor: CLIP 0.68 vs a random-ViT readout 0.54 vs chance 0.50, so +0.14 is real learned orthographic signal, not the readout overfitting. No generation required."
  },
  "inputs": [
    {"type": "image", "example": "a natural photograph", "benchmark": "MajajHong2015 V4 / IT", "desc": "Still images drive the ventral stream; predicted V4/IT responses map onto occipitotemporal cortex."},
    {"type": "text", "example": "a sentence", "benchmark": "Pereira2018", "desc": "Sentences drive the language network; a causal LM's features predict left frontotemporal language regions."},
    {"type": "video", "example": "a 3-second clip", "benchmark": "Lahner2024 BOLDMoments", "desc": "Short videos drive dorsal + ventral visual cortex; native-temporal V-JEPA leads."},
    {"type": "video + audio", "example": "a movie segment", "benchmark": "Algonauts2025 CNeuroMod", "desc": "Continuous movies drive much of cortex; banded ridge over video+audio+text beats any single stream."}
  ]
};
