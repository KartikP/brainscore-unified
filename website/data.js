// Inlined results so the site opens from file:// without fetch/CORS issues.
// Canonical copy lives in data/results.json (identical content).
window.BSU_DATA = {
  "meta": {
    "title": "Brain-Score · Unified Model Interface",
    "subtitle": "Register a model once. Score it across vision, language, audio, video, multimodal, perturbation, and embodied benchmarks — through one process() interface.",
    "note": "Matched nulls are DEFINED for every capability; floors are MEASURED for neural encoding, behavior, and (on real BOLD) temporal alignment. Where a curve is non-monotonic, a result is contingent on a scoring choice, or a demo only proves plumbing, we say so — see each reading and the limitations panel.",
    "provenance": "Neural/behavioral scores reproduced bit-for-bit on v1.5 from the v1 baselines; embodied scores from scripts/vlm_game; the temporal-shift null was run on real 162k-TR Algonauts BOLD; figures from brainscore.visualization. Numbers carried from v1 are labelled; synthetic illustrative values are labelled as such."
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
      "Encoding scores are raw Pearson r without per-voxel noise-ceiling normalization; point-estimate gaps lack bootstrap CIs."
    ]
  },
  "nulls": {
    "description": "Matched nulls are DEFINED for every capability and run through the same pipeline with the signal destroyed in one specific way. Floors are MEASURED for neural encoding and behavior; the temporal-shift null was run on real 162k-TR Algonauts BOLD (shuffle floor ≈ 0.012, score peaks at the true HRF delay and collapses when mis-timed).",
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
    "capability": "Induced dyslexia via process(StateChange) — Qwen2.5-VL-3B on ROAR",
    "conditions": {
      "baseline": [0.93, 0.93, 0.94],
      "lesioned": [0.66, 0.68, 0.65],
      "random control": [0.91, 0.90, 0.92],
      "restored": [0.93, 0.93, 0.94]
    },
    "chance": 0.5,
    "reading": "Ablating the pseudo-selective MLP population (K=500/layer) drops accuracy from 1.00 to 0.67 — across the 0.65 dyslexia threshold. A random same-size ablation barely moves it. reset() restores bit-for-bit."
  },
  "selection": {
    "capability": "Composite selection — units across layers for one region",
    "layers": ["blocks.5", "blocks.10", "blocks.16", "blocks.20"],
    "selected_counts": [3, 50, 120, 18],
    "units_per_layer": 1024,
    "reading": "A CompositeSelector gathers a functional population spanning depth into one region; the readout draws most from the middle-late layers."
  },
  "inputs": [
    {"type": "image", "example": "a natural photograph", "benchmark": "MajajHong2015 V4 / IT", "desc": "Still images drive the ventral stream; predicted V4/IT responses map onto occipitotemporal cortex."},
    {"type": "text", "example": "a sentence", "benchmark": "Pereira2018", "desc": "Sentences drive the language network; a causal LM's features predict left frontotemporal language regions."},
    {"type": "video", "example": "a 3-second clip", "benchmark": "Lahner2024 BOLDMoments", "desc": "Short videos drive dorsal + ventral visual cortex; native-temporal V-JEPA leads."},
    {"type": "video + audio", "example": "a movie segment", "benchmark": "Algonauts2025 CNeuroMod", "desc": "Continuous movies drive much of cortex; banded ridge over video+audio+text beats any single stream."}
  ]
};
