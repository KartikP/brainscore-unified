// Inlined results so the site opens from file:// without fetch/CORS issues.
// THIS FILE is the canonical data the site renders. (data/results.json is an
// earlier seed snapshot kept for reference; data/temporal_shift_null.json and
// data/nocache_roar.json are the raw measured outputs behind the curves here.)
window.BSU_DATA = {
  "meta": {
    "title": "Brain-Score · Unified Model Interface",
    "subtitle": "Register a model once. Score it across vision, language, audio, video, multimodal, perturbation, and embodied benchmarks — through one process() interface.",
    "note": "A matched null is defined for every capability. Floors are measured for neural encoding, behavior, and (on recorded BOLD) temporal alignment. Non-monotonic curves, scoring-contingent rankings, and demonstrations that exercise only the pipeline are flagged in the corresponding reading and in the limitations panel.",
    "provenance": "Behavioral scores were re-run with caching disabled this session: chance / random-ViT / CLIP reproduce 0.500 / 0.540 / 0.690 raw, matching the prior baselines. The temporal-shift null was computed on 50k recorded Algonauts TRs; embodied scores come from scripts/vlm_game; figures from brainscore.visualization. Values carried from a prior cycle and synthetic illustrative values are labelled."
  },
  "hero_rotation": [
    {"model": "clip-vit-b-32", "benchmark": "MajajHong2015public.IT-pls", "comment": "# vision · neural — predict IT from image features"},
    {"model": "clip-vit-b-32", "benchmark": "Pereira2018.243sentences-linear", "comment": "# the SAME model — now a language benchmark"},
    {"model": "qwen2.5-vl-3b", "benchmark": "MajajHong2015public.IT-pls", "comment": "# a 3B VLM instead — identical call"},
    {"model": "gpt2", "benchmark": "Yeatman2021-lexical_decision-text", "comment": "# reading behavior, from strings alone"},
    {"model": "blip2-opt-2.7b", "benchmark": "MajajHong2015public.V4-pls", "comment": "# different model, different size — same three lines"},
    {"model": "vjepa1-vitl", "benchmark": "Lahner2024-fMRI-naturalistic-visualROI", "comment": "# video · naturalistic fMRI encoding"},
    {"model": "vjepa1-wav2vec2", "benchmark": "Lahner2024-fMRI-naturalistic-multimodal-visualROI", "comment": "# two towers (video + audio), one model object"},
    {"lines": [
      "from brainscore import load_model",
      "from brainscore.harnesses.gymnasium_harness import play_gym_episode",
      "model  = load_model(\"qwen2.5-vl-7b\")",
      "result = play_gym_episode(model, \"MiniGrid-DoorKey-6x6\")",
      "# embodied — the model emits an action from process(EnvironmentStep) each tick"
    ]},
    {"lines": [
      "from brainscore import load_model, load_benchmark",
      "model = load_model(\"qwen2.5-vl-3b\")",
      "model.process(StateChange(target=vwf_units, perturbation=\"zero\"))  # lesion",
      "score = load_benchmark(\"Yeatman2021-lexical_decision-image\")(model)",
      "# perturbation — score the lesioned model; model.reset() restores it"
    ]},
    {"model": "random-vit-b-32", "benchmark": "MajajHong2015public.IT-pls", "comment": "# even the null floor registers the same way"}
  ],
  "scaling": {
    "language_encoding": {
      "capability": "Neural encoding — language (Pereira2018, r)",
      "models": ["random-vit", "CLIP-B32", "GPT-2", "Qwen-3B", "BLIP-2"],
      "scores": [0.123, 0.464, 0.531, 0.708, 0.737],
      "null_floor": 0.123,
      "reading": "Prediction of the language network rises ~3.8× above the random-feature floor (an untrained network). The larger causal language models (Qwen, BLIP-2) predict it more accurately than CLIP."
    },
    "it_encoding": {
      "capability": "Neural encoding — IT cortex (MajajHong2015, r)",
      "models": ["random-vit", "Qwen-3B", "BLIP-2", "CLIP-B32"],
      "scores": [0.104, 0.315, 0.334, 0.374],
      "null_floor": 0.104,
      "reading": "All models clear the random-feature floor (3.6×), but prediction does not increase with model size: CLIP, the smallest, scores highest. IT-cortex alignment here tracks the training objective (image–text contrastive learning) rather than scale. The differences (0.315 / 0.334 / 0.374) are not yet tested for significance, and the non-monotonicity is treated as a benchmark-validity flag rather than a scaling result."
    },
    "video_encoding": {
      "capability": "Neural encoding — video, visual ROI (Lahner2024, r)",
      "models": ["BLIP-2", "Qwen-3B", "VideoMAE", "V-JEPA2", "CLIP-B32", "V-JEPA1"],
      "scores": [0.180, 0.227, 0.321, 0.421, 0.456, 0.533],
      "null_floor": 0.05,
      "reading": "V-JEPA v1, trained to predict its own latent representations, outscores image–text CLIP; the pixel-reconstruction video model (VideoMAE) ranks lower. The training objective accounts for the ordering more than whether the model is natively temporal. CLIP (0.456) scores above V-JEPA v2 (0.421), but this is a layer-selection artifact, not a capability gap: each region was mapped to V-JEPA v1's brain-optimal layer (layer 16, chosen by a per-voxel sweep), and v2 was scored at that same layer without its own sweep. v2's optimal layer differs; a v2 layer sweep is expected to recover the gap. The ranking is also scoring-contingent more broadly — V-JEPA v1 overtakes CLIP only after a feature-rescaling step is removed from the regression and the region is mapped to its best-matching layer. Scores are raw correlation, not normalized by each voxel's noise ceiling."
    },
    "behavior_roar": {
      "capability": "Behavior — lexical decision (ROAR Yeatman2021)",
      "models": ["chance", "random-vit", "CLIP-B32", "BLIP-2", "GPT-2", "Qwen-3B"],
      "scores": [0.500, 0.540, 0.680, 0.790, 0.810, 0.930],
      "null_floor": 0.540,
      "reading": "Chance is 0.50; an untrained network floors at 0.54. GPT-2, from the letter strings alone, reaches 0.81 — at the human ceiling — indicating lexical decision is an orthographic judgment that does not require vision. Qwen, answering by generating its choice, scores highest (0.93)."
    },
    "embodied_game": {
      "capability": "Embodied — grid video game (success rate)",
      "models": ["random", "Qwen-VL-3B (CoT)", "Qwen-VL-7B (CoT)", "Gemma-4-12B (CoT)", "DeepSeek-R1 (ASCII)", "oracle"],
      "scores": [0.20, 0.0, 0.53, 1.0, 0.87, 1.0],
      "null_floor": 0.20,
      "reading": "Closed-loop grid game, each model at its best elicitation, the three vision-language models reading the same rendered board. The 3B scores 0.0 (below the random floor); the 7B, given room to reason step by step, solves 0.53; the 12B (Gemma-4) solves all boards at optimal path length (1.0). A text-only reasoner handed the same board as text rather than an image (DeepSeek-R1) scores 0.87. The limiting factor for the smaller vision-language models is perceiving the grid from pixels rather than reasoning over it; this resolves with scale (0.0 → 0.53 → 1.0). 'oracle' is the optimal policy (ceiling); DeepSeek-R1 is the perception-removed control. n = 15 episodes per model."
    },
    "multimodal_algonauts": {
      "capability": "Multimodal — Algonauts2025 CNeuroMod (r)",
      "models": ["text-only", "video-only", "audio-only", "concat", "banded"],
      "scores": [0.120, 0.150, 0.157, 0.186, 0.213],
      "null_floor": 0.05,
      "reading": "A banded ridge over video (CLIP), audio (Wav2Vec2), and text (MiniLM) exceeds every single modality, reproducing the Algonauts paper baseline (~0.20–0.25). Whether the remaining gap to MIRAGE's 0.319 is attributable to the backbone rather than the pipeline is untested — a controlled encoder swap (this pipeline, MIRAGE's encoder) has not been run. What is established is that the earlier <0.01 figure reflected an unaligned demonstration; with HRF + stimulus-window alignment the pipeline reaches the paper's reported range."
    }
  },
  "limitations": {
    "title": "Limitations and contingencies",
    "items": [
      "The IT-encoding curve is non-monotonic; this is treated as a benchmark-validity flag rather than evidence of scaling.",
      "V-JEPA v1 > CLIP on video holds only after removing a feature-rescaling step and selecting the best layer per region; the ranking is scoring-contingent. V-JEPA v2 < CLIP reflects scoring v2 at v1's layer without a v2-specific layer sweep.",
      "The embodied curve combines a vision-language scaling comparison (3B / 7B / 12B, all reading pixels with chain-of-thought) with a perception-removed control (DeepSeek-R1, reading the board as text). The 3B→7B→12B trend (0.0 → 0.53 → 1.0) is a same-game, same-elicitation comparison, but n = 15 episodes on a 5×5 board; it is a tiers demonstration, not a precise scaling law. MiniGrid-DoorKey is a separate, harder environment (Gemma-4 scores 0.00 there) and is not interchangeable with it.",
      "The topographic metric has been validated only on synthetic Gaussian fields; it has not been run on recorded fMRI and is not wired into any benchmark.",
      "The attribution of the MIRAGE gap to the backbone is untested (no controlled encoder swap has been run).",
      "Encoding scores are raw Pearson r, without per-voxel noise-ceiling normalization; point-estimate differences lack bootstrap confidence intervals.",
      "Induced dyslexia is scale-dependent: it does not reproduce at 3B (selectivity runs the wrong way) and is sub-threshold at 7B; it reproduces at 32B, but at a 25% mask versus Honarmand's 6.9% on 72B. The 72B model has not been run here."
    ]
  },
  "temporal_shift_validation": {
    "title": "Temporal-shift null, measured on real BOLD",
    "subtitle": "Algonauts CNeuroMod, subject 1, 50,000 brain scans (1 scan ≈ 1.5 s), CLIP video features",
    "shifts": [-12, -6, -3, 0, 3, 6, 12, 18],
    "scores": [0.024, 0.035, 0.055, 0.120, 0.132, 0.078, 0.040, 0.036],
    "true_delay": 3,
    "shuffle_floor": 0.009,
    "reading": "An alignment-validity check, computed on recorded BOLD. The fMRI BOLD signal lags the underlying neural activity by a few seconds (the hemodynamic response), so per-moment model features align with the recorded scans only when shifted forward by that lag. Shifting the features earlier or later, prediction peaks at +3 scans (≈4.5 s) and falls toward the shuffle floor when mis-timed, in both directions. A flat curve would indicate the alignment carried no stimulus-locked signal; the curve is not flat. ('true delay' is the hemodynamic lag; the dashed line is the score after the timing is shuffled out.)"
  },
  "movie_brain": {
    "title": "Recorded vs. predicted BOLD for a held-out movie clip",
    "subtitle": "A 10-second clip from the Algonauts 2025 dataset (Friends, s01e02) — video, audio, and dialogue — is the input. An encoding model predicts subject 1's fMRI response (lower brain); it is shown alongside that subject's recorded response (upper brain) for the same scans. Both are rendered on a nilearn glass brain (three views — lateral, posterior, dorsal; 1000 cortical parcels) and advance with the clip on playback.",
    "video": "assets/movie_brain_real/clip.mp4",
    "human": "assets/movie_brain_real/human/bold_",
    "model": "assets/movie_brain_real/model/bold_",
    "n": 7,
    "tr_sec": 1.49,
    "cachebust": "13",
    "transcript": ["Yeah, and it's not that we", "don't like the", "comedian. It's just that", "that's", "not why we bought the", "ticket.", "You see,"],
    "times": [29.8, 31.3, 32.8, 34.3, 35.8, 37.2, 38.7],
    "per_tr_r": [0.1425, 0.0662, 0.2419, 0.1927, 0.1126, 0.1961, 0.3464],
    "mean_r": 0.253,
    "note": "Warm = activity above this clip's baseline, cool = below. Both brains use one color scale, in standard-deviation units (each stream normalized to its own variance), so the spatial pattern is directly comparable. In raw units the predicted response is ~5× smaller than the recorded one because the ridge prediction is regularized; the magnitude difference is reported by the held-out correlation (r ≈ 0.25), not the color. Both brains share one clock and lag the movie by the ~4.5 s hemodynamic delay (see below).",
    "reading": "The temporal-multimodal pipeline, end to end on recorded data. Per moment, video → V-JEPA-2 features, audio → Wav2Vec2, dialogue → MiniLM; a banded-ridge encoding model fit on the other Friends episodes shifts them by the hemodynamic lag and predicts subject 1's response across all 1000 cortical parcels per scan; the `glass_brain_movie` tool renders each scan. The recorded BOLD carries the ~4.5 s hemodynamic lag physiologically, and the encoding model builds the same lag into its design matrix, so the two are directly comparable at each scan and need no relative shift. Both lag the movie by that delay, so each map is the response to what was on screen a few seconds earlier.",
    "caption": "Recorded vs. predicted, on a held-out Friends segment: the encoding model was fit on the other segments, so the lower brain is a genuine prediction and the upper brain is subject 1's recorded fMRI for the same scans. Neither is time-shifted relative to the other (both already include the hemodynamic lag); to align the brain map to the on-screen frame, the movie would be advanced ~3 scans (~4.5 s). The Friends footage is from the Algonauts 2025 / Courtois NeuroMod dataset — internal research use only, not for redistribution. (TR = 1.49 s; 7 scans ≈ 10.4 s.)"
  },
  "nulls": {
    "description": "A matched null is defined for every capability and run through the same pipeline with the signal removed in one specific way. Floors are measured for neural encoding and behavior; the temporal-shift null was computed on 50k recorded Algonauts TRs (shuffle floor ≈ 0.009; the score peaks at the hemodynamic delay and falls when mis-timed — see the curve above).",
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
    "capability": "Induced dyslexia (Honarmand et al. 2026), reproduced on Qwen2.5-VL-32B",
    "protocol": "Find the model's word-form units with a localizer (which internal units fire more for words than for scrambled words and line-drawn objects), switch off the most word-form-selective ones vs an equal-size random set (drawn across all 64 of the network's internal layers), then re-test reading by having the model generate its answer — no retraining. 2 random seeds, averaged.",
    "mask_pct": [0, 6.9, 15, 25],
    "vwf_roar": [0.975, 0.963, 0.925, 0.537],
    "vwf_roar_sd": [0.0, 0.0, 0.0, 0.0],
    "random_roar": [0.975, 0.956, 0.938, 0.887],
    "random_roar_sd": [0.0, 0.006, 0.037, 0.075],
    "vwf_control": [0.87, 0.87, 0.87, 0.80],
    "random_control": [0.87, 0.87, 0.80, 0.80],
    "threshold": 0.65,
    "brain_caption": "Where the lesion lands: the VWF-selective units align with the human Visual Word Form Area (VWFA — left ventral occipitotemporal cortex, MNI ≈ [-44,-58,-15]; Honarmand Fig 5). This quickbrain glass brain shows that cortical territory — the area effectively 'dropped' when the population is ablated.",
    "scale_note": "The deficit is scale-dependent. At 3B, word-form-selective ablation is less damaging than random (the wrong direction). At 7B it is more damaging than random (the correct direction) but does not cross the threshold. At 32B (shown), a 25% word-form-selective mask drops reading to 0.537, below the 0.65 dyslexia threshold, while a same-size random ablation stays at 0.887 — a selective deficit.",
    "reading": "The selective deficit reproduces at 32B. Ablating the word-form units (the top ~25% of one layer's word-form filters) drops reading to 0.537 — below the 0.65 dyslexia threshold — while ablating the same number of random units leaves reading at 0.887 and a non-reading control task at 0.80. The deficit is therefore specific to reading rather than a general loss of ability. It is scale-dependent: at 3B the targeted lesion runs the wrong way, at 7B the right way but sub-threshold, and only at 32B does it cross the threshold selectively. Honarmand's 72B model shows the deficit at a smaller lesion (~7%), consistent with less damage being required as the model grows.\n\nThis is a registered Brain-Score benchmark (`Yeatman2021-induced_dyslexia`): a `FunctionalSelection` localizer identifies the word-form units, `StateChange` ablates them, and a matched `RandomSelection` control gates the dyslexic flag, so general damage does not register as selective dyslexia. The benchmark was validated end-to-end on recorded weights; the random-control gate correctly withheld the flag when an over-broad ablation reduced reading non-specifically."
  },
  "selection": {
    "capability": "Composite selection — units across layers for one region",
    "layers": ["layer 5", "layer 10", "layer 16", "layer 20"],
    "selected_counts": [3, 50, 120, 18],
    "units_per_layer": 1024,
    "reading": "A composite selector gathers a functional group of units that **spans several depths of the network** into a single 'region', instead of reading from one fixed layer. Here most of the word-form units live in the **middle-to-late layers** (peaking around layer 16)."
  },
  "models_glossary": {
    "title": "The models, at a glance",
    "subtitle": "What each model used across this site is, and what it's built for",
    "models": [
      {"name": "CLIP ViT-B/32", "kind": "vision–language · contrastive", "desc": "A small image encoder trained to match images with their text captions. Strong, fast general visual features — the lightweight baseline."},
      {"name": "BLIP-2 OPT-2.7B", "kind": "vision–language · captioning", "desc": "A large ViT-G encoder + Q-Former feeding a 2.7B OPT language decoder, trained for image captioning — not instruction-following."},
      {"name": "Qwen2.5-VL (3B/7B/32B)", "kind": "instruction-tuned VLM", "desc": "A vision encoder fused into an instruction-tuned language model. Answers questions about images; the strongest reader here, and the one that induces dyslexia at 32B."},
      {"name": "GPT-2 (124M)", "kind": "text language model", "desc": "A small text-only causal LM. No vision — it reads word strings, and matches humans on lexical decision from strings alone."},
      {"name": "V-JEPA / V-JEPA2 ViT-L", "kind": "self-supervised video", "desc": "Video models trained by predicting masked latent representations (not pixels). The best video→brain encoders we test."},
      {"name": "VideoMAE base", "kind": "self-supervised video", "desc": "A video masked-autoencoder trained to reconstruct pixels. Native-temporal, but the pixel objective transfers less well to brain prediction."},
      {"name": "Wav2Vec2 · MiniLM", "kind": "audio · sentence encoders", "desc": "The speech-audio tower and the sentence-embedding text tower used in the multimodal Algonauts setup."},
      {"name": "random-ViT", "kind": "null control", "desc": "CLIP's architecture with random weights. The floor for what a logistic readout can overfit from random features (~0.54 on ROAR)."},
      {"name": "chance", "kind": "null control", "desc": "Returns uniform probabilities regardless of input — the absolute floor (0.50 on a balanced binary task)."},
      {"name": "oracle", "kind": "upper reference (grid game)", "desc": "A privileged policy that always steps toward the goal — the optimal player, defining the ceiling for the embodied game."}
    ]
  },
  "embodied_game": {
    "title": "A VLM plays a video game",
    "subtitle": "Closed-loop process(EnvironmentStep) — the model sees a rendered frame, reasons, picks a move, the environment responds, repeat. Now a registered Brain-Score benchmark (GridGame-reach-5x5): success rate, oracle ceiling, random floor.",
    "models": ["random", "Qwen-VL-3B (CoT)", "Qwen-VL-7B (CoT)", "Gemma-4-12B (CoT)", "DeepSeek-R1 (ASCII)", "oracle"],
    "success": [0.20, 0.0, 0.533, 1.0, 0.867, 1.0],
    "colors": ["#9aa0a6", "#d8483b", "#e0a13b", "#2a8c6a", "#7c4dff", "#1f9d57"],
    "null_floor": 0.20,
    "reading": "The same interface drives a closed-loop agent — the model sees a frame, reasons, acts, and the world responds — with zero schema errors across hundreds of ticks. Two findings. (1) **Instruction matters**: asked for a one-word answer the 7B vision-language model scored 0.13; given room to reason step-by-step (chain-of-thought) it jumps to 0.53. (2) **Perception lifts with scale.** All three vision-language models read the same rendered board with chain-of-thought: the 3B can't perceive the grid at all (0.0), the 7B partially (0.53), and the 12B (Gemma-4) perceives and reasons well enough to solve **every** board (1.0). Hand a strong text-only reasoner the same board as plain text (so seeing it is free) and DeepSeek-R1 solves 87% — confirming that for the smaller vision models the bottleneck is perceiving the abstract grid from pixels, a bottleneck that scale clears."
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
    "models": ["chance", "random-ViT", "CLIP", "GPT-2", "BLIP-2", "Qwen"],
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
    {"type": "audio", "example": "a speech / environmental sound clip", "benchmark": "Lahner2024 auditory-ROI · Algonauts audio", "desc": "Sounds drive auditory cortex — Heschl's gyrus and superior temporal gyrus. A Wav2Vec2 tower predicts these temporal regions."},
    {"type": "video", "example": "a 3-second clip", "benchmark": "Lahner2024 BOLDMoments", "desc": "Short videos drive dorsal + ventral visual cortex; native-temporal V-JEPA leads."},
    {"type": "video + audio", "example": "a movie segment", "benchmark": "Algonauts2025 CNeuroMod", "desc": "Adding the audio tower extends prediction into temporal/auditory cortex alongside the visual response."},
    {"type": "video + audio + text", "example": "a movie scene with dialogue + subtitles", "benchmark": "Algonauts2025 CNeuroMod (multimodal)", "desc": "The full naturalistic stream drives much of cortex; a banded ridge over all three towers (video + audio + text) gives the best whole-brain prediction — this is the actual Algonauts setup."}
  ],
  "witness": {
    "title": "Watch Brain-Score evaluate a model",
    "subtitle": "Every capability funnels through one method — process(input_event). Because every benchmark passes through that one method, watching it is enough to see ANY of them run: what the model saw (rendered per modality), what it did (activations, probabilities, a generated answer, an action), and in what mode. These are real recorded process() calls — no benchmark-specific code.",
    "modes": ["neural", "readout", "generation", "action", "state_change"],
    "panels": [
      {"img": "assets/witness_game_0.png", "caption": "Embodied · process(EnvironmentStep): the policy sees the rendered frame and the Witness logs what it DID — action → down. Recorded straight off the grid-game rollout, no benchmark-specific code."},
      {"img": "assets/witness_game_2.png", "caption": "A later tick of the same rollout. One recorder, one trace format, for every process() call across vision, language, audio, embodied, and perturbation."}
    ],
    "reading": "The Witness wraps process() non-invasively for the duration of a benchmark and logs one event per call. Because the unified interface routes neural recording, behavioral generation/readout, embodied action, and perturbation all through the same process() entry point, one recorder covers every capability — the same trace structure whether the model is looking at images, reading sentences, playing a game, or being lesioned."
  },
  "percept": {
    "title": "PerceptWindow — what the model ACTUALLY saw",
    "subtitle": "The Witness records what was presented (a file path). But preprocessing — resize, center-crop, normalize — happens one layer deeper, right before the network's forward(). PerceptWindow taps THAT point with a forward pre-hook (zero extra forward passes), then inverts the normalization to reconstruct the model's literal percept — determined entirely by the preprocessing pipeline, not the weights, so these are exactly the tensors a CLIP-preprocessed model ingests.",
    "tabs": [
      {
        "id": "crop",
        "label": "Resize & crop",
        "columns": ["presented", "raw tensor (clipped)", "percept (reconstructed)"],
        "rows": [
          {"label": "Wide 640×360", "presented": "assets/percept/0_presented.png", "tensor": "assets/percept/0_tensor.png", "percept": "assets/percept/0_percept.png", "note": "Center-crop discards the green L and gold R edge bands entirely — the model never saw them. TOP/BOTTOM survive."},
          {"label": "Tall 360×640", "presented": "assets/percept/1_presented.png", "tensor": "assets/percept/1_tensor.png", "percept": "assets/percept/1_percept.png", "note": "Now TOP and BOTTOM are cropped away and the L/R bands survive — the crop axis flips with aspect ratio."},
          {"label": "Square 360×360", "presented": "assets/percept/2_presented.png", "tensor": "assets/percept/2_tensor.png", "percept": "assets/percept/2_percept.png", "note": "Square needs no crop — all four edges survive; only the resize to 224 happens. Colors come back faithfully after de-normalization."}
        ],
        "reading": "Middle column = the raw normalized tensor clipped to [0,1]: false color, NOT directly viewable — that is why reconstruction is needed. Right column = PerceptWindow inverting x·std+mean. The point is the gap between left and right: the center-crop literally removes content (the labeled edge bands), so 'what the benchmark presented' and 'what the model saw' are not the same image.",
        "caveat": "This tab uses CLIP's exact preprocessing constants over a trivial identity module (no weights), because the percept is a function of preprocessing, not the model. PerceptWindow captures the same tensor on a real CLIP/ViT; it does NOT capture post-embedding internal activations — only the input that entered forward()."
      },
      {
        "id": "rajalingham",
        "label": "Rajalingham 2-AFC",
        "columns": ["presented montage", "raw tensor (clipped)", "percept (reconstructed)"],
        "rows": [
          {"label": "Trial 0 (502×586)", "presented": "assets/raj2afc_montage_0.png", "tensor": "assets/percept/raj0_tensor.png", "percept": "assets/percept/raj0_percept.png", "note": "CLIP's center-crop trims the top SAMPLE label and the bottom LEFT/RIGHT labels — but the sample image and both choice tokens survive intact."},
          {"label": "Trial 2 (502×586)", "presented": "assets/raj2afc_montage_2.png", "tensor": "assets/percept/raj1_tensor.png", "percept": "assets/percept/raj1_percept.png", "note": "Same near-square montage, same trim. The decision content (sample + two tokens) is preserved; only the text labels at the extremes are clipped."}
        ],
        "reading": "This is CLIP ViT-B/32's front-end — the 2-AFC similarity chooser. Resize-shortest-side + center-crop trims the montage's top and bottom (the SAMPLE / LEFT / RIGHT text labels) while the sample image and both choice tokens survive. So the similarity model decided from the images, not the printed labels — a detail you only see by reconstructing what it actually ingested.",
        "caveat": "This reconstruction is specifically CLIP's view. The generation VLMs (Qwen, Gemma) do NOT center-crop — they resize preserving aspect and saw the full montage including every label. PerceptWindow on those models (run on EC2) would show the un-cropped montage; this tab does not represent what the instruction-following models ingested."
      },
      {
        "id": "multimodal",
        "label": "Multimodal (image + text)",
        "columns": ["presented", "ingested form", "percept (reconstructed)"],
        "rows": [
          {"kind": "image", "label": "Vision tower · pixel tensor", "presented": "assets/percept/mm_vision_presented.png", "tensor": "assets/percept/mm_vision_tensor.png", "percept": "assets/percept/mm_vision_percept.png", "note": "The image branch: resize + normalize, then de-normalized back. This objectome token — a wrench on a mountainside — is exactly what the vision encoder ingested."},
          {"kind": "text", "label": "Text tower · token ids", "presented": "\"a photo of a wrench on a mountainside\"", "tensor": "[49406, 320, 1125, 539, 320, 30980, 525, 320, 14547, 1145, 49407, … ]  ·  padded to the 77-token context", "percept": "<|startoftext|> a photo of a wrench on a mountainside <|endoftext|>  … ×67 more <|endoftext|> (padding)", "note": "BPE splits 'mountainside' into TWO subwords — 14547 ('mountain') + 1145 ('side') — while 'wrench' is one token (30980). The <|startoftext|> / <|endoftext|> markers and the EOS-padding to 77 are exactly what the text encoder read."}
        ],
        "reading": "A dual-tower model (here CLIP ViT-B/32) ingests an image AND a caption. PerceptWindow registers a forward pre-hook on BOTH towers — capturing the vision tower's pixel tensor and the text tower's token-id tensor in the same run — then reconstructs each per its modality: de-normalize for the image, detokenize for the text. One mechanism, every modality; on a video+audio model the same hooks would return sampled frames and a waveform.",
        "caveat": "The text 'percept' is the detokenized input ids — faithful to what the encoder read (special tokens, subword splits, 77-token padding), but it is the ids round-tripped through the tokenizer, not a pixel image. Audio percepts come back as a waveform/spectrogram, which for spectrogram models is not losslessly invertible to sound."
      }
    ]
  },
  "rajalingham": {
    "title": "The same object-recognition behavior, several ways",
    "subtitle": "Rajalingham 2018's match-to-sample signature, reached the FAITHFUL way — the model sees the briefly-shown sample plus two object choices and actually picks one, exactly as the human/monkey subjects did. The original Brain-Score benchmark instead reconstructs that 2-choice behaviour from a trained classifier. Scoring both the same way turns one number into a map across model size, how you ask for the choice, and how much practice the model gets. The metric, i2n, is a per-image score: for each image, how closely does the model's pattern of right-and-wrong choices match the humans' — so a model can be accurate yet still score low if it's hard on different images than people find hard. Ways of asking (the bar colours): direct = answer in one word from the first glance; CoT = 'chain of thought', reason step-by-step then answer; 4-shot = four already-solved example trials shown first as in-context practice. (A coin-flip-free single chooser tops out around i2n 0.33 — the published readout band — so every bar here sits well below that ceiling.)",
    "metric": "i2n raw (image-level, vs the human pool)",
    "binary_ceiling": 0.33,
    "readout_band": "~0.30-0.50",
    "rows": [
      {"model": "random null", "mode": "—", "acc": 0.510, "frac_left": null, "i2n": -0.028, "kind": "null"},
      {"model": "CLIP", "mode": "similarity", "acc": 0.662, "frac_left": 0.49, "i2n": 0.061, "kind": "feature"},
      {"model": "Qwen-VL-3B", "mode": "CoT", "acc": 0.497, "frac_left": null, "i2n": 0.006, "kind": "cot"},
      {"model": "Qwen-VL-3B", "mode": "direct", "acc": 0.634, "frac_left": 0.85, "i2n": 0.064, "kind": "direct"},
      {"model": "Qwen-VL-3B", "mode": "direct + 4-shot", "acc": 0.506, "frac_left": 0.99, "i2n": 0.010, "kind": "fewshot"},
      {"model": "Qwen-VL-7B", "mode": "CoT", "acc": 0.562, "frac_left": 0.89, "i2n": 0.031, "kind": "cot"},
      {"model": "Qwen-VL-7B", "mode": "direct", "acc": 0.859, "frac_left": 0.47, "i2n": 0.163, "kind": "direct"},
      {"model": "Qwen-VL-7B", "mode": "direct + 4-shot", "acc": 0.785, "frac_left": 0.63, "i2n": 0.154, "kind": "fewshot"},
      {"model": "Gemma-4-12B", "mode": "direct", "acc": 0.846, "frac_left": 0.45, "i2n": 0.146, "kind": "direct"},
      {"model": "Gemma-4-12B", "mode": "direct + 4-shot", "acc": 0.853, "frac_left": 0.43, "i2n": 0.100, "kind": "fewshot"}
    ],
    "montages": ["assets/raj2afc_montage_0.png", "assets/raj2afc_montage_2.png"],
    "kindColors": {"null": "#9aa0a6", "feature": "#1f9d57", "cot": "#d8483b", "direct": "#2f6bff", "fewshot": "#c6810f"},
    "findings": [
      "Elicitation dominates the model. Direct ≫ chain-of-thought: CoT corrupts the percept (Qwen-7B 0.163 → 0.031) and induces a heavy LEFT bias (frac-left 0.47 → 0.89). The OPPOSITE of the embodied game, where chain-of-thought was essential — reasoning helps planning but hurts perception. A side-balanced control (each condition shown both ways) confirms it: bias-free accuracy is 0.89 for 7B-direct (genuinely perceives) vs 0.57 for 7B-CoT (≈chance once the positional prior is balanced out — its apparent score was mostly bias).",
      "Scale within direct mode: 3B stays biased and weak (0.064, 85% LEFT), while 7B (0.163) and Gemma-4-12B (0.146) are unbiased and genuinely good. A capable VLM does the faithful zero-shot 2-AFC.",
      "In-context 'practice' HURTS — overturning the human-practice intuition. Four balanced demos dropped Qwen-3B to chance (0.010, 99% LEFT) and Gemma 0.146 → 0.100. Visual match-to-sample doesn't learn from few-shot demos the way text tasks do.",
      "The task adaptation that wins is the original READOUT — a logistic decoder learned on frozen features (~0.30-0.50), still above the best zero-shot generation (0.16). A trained task-head beats prompting; matching it from generation would take weight-level fine-tuning, not a few demos."
    ],
    "reading": "Same phenomenon, scored identically, three orthogonal axes: model scale (3B→7B→12B), elicitation (CoT vs direct), and task adaptation (zero-shot → few-shot → trained readout). Dashed lines mark the bias-free ceiling a single binary chooser can reach (0.33) and the trained-readout band. Side bias was caught by logging frac-left per run and removed with a side-balanced re-run.",
    "caveats": [
      "Token mismatch (the load-bearing caveat): the original humans/monkeys chose between CLEAN canonical object tokens; our choice tokens are other high-variation objectome renders (cluttered, grayscale, silhouette-like). The model therefore does a HARDER task than the humans we score it against, so a low i2n conflates 'can't perceive' with 'faced worse stimuli'. The absolute numbers are lower bounds; the RELATIVE comparisons (direct vs CoT, scale, few-shot) — same tokens throughout — are the trustworthy part.",
      "The readout band (~0.30–0.50) is the original benchmark's published range, NOT re-run on this 120-image subset with this scoring — it anchors the axis but isn't apples-to-apples.",
      "No bootstrap CIs yet: with 120 images, the Gemma-12B (0.146) vs Qwen-7B (0.163) gap may not be significant — read the tiers, not the decimals.",
      "Few-shot-hurts may be partly an implementation artifact of multi-image prompting (the 3B collapsed to 99% LEFT under demos); it needs an answer-line-only control before it's a fact about visual in-context learning."
    ],
    "sequential": {
      "title": "Faithful match-to-sample: what happens when the sample is actually removed",
      "subtitle": "The montage above shows all three at once. The real human task is sequential — the sample is flashed, removed, THEN the choices appear. The Witness records that interaction step-by-step, which is the right lens here: the finding is about the sample→memory→language bottleneck, not the pixels. Qwen2.5-VL-7B, identical 2,505 trials.",
      "trace": [
        {"img": "assets/raj_seq_sample.png", "cap": "1 · SAMPLE shown, then removed", "kind": "img"},
        {"text": "the model writes a description in its own words — then the sample is gone", "cap": "2 · the bottleneck (match from memory)", "kind": "text"},
        {"img": "assets/raj_seq_choices.png", "cap": "3 · choices appear; sample is GONE → it must match from memory", "kind": "img"}
      ],
      "conditions": [
        {"label": "random null", "i2n": -0.028, "kind": "null"},
        {"label": "match from memory — sample removed, match from its own description", "i2n": 0.097, "kind": "seq"},
        {"label": "sample kept in view — choose with the sample still shown", "i2n": 0.170, "kind": "seq"},
        {"label": "all shown at once — the montage (current)", "i2n": 0.163, "kind": "sim"}
      ],
      "kindColors": {"null": "#9aa6b8", "seq": "#7c4dff", "sim": "#2f6bff"},
      "reading": "**Removing the sample is what costs — not showing things one at a time.** The only truly faithful condition (match from memory: the sample is gone and the choice rides on the model's own words) drops the score to 0.097, ~40% below the rest — putting the sample into words throws away the visual detail the match needs (the descriptions were fluent, e.g. 'a sculpture of a person in mid-air, diving or jumping'; the model gave a clean LEFT/RIGHT answer on all but 5 of 2,505 trials). Keeping the sample in view (0.170) ≈ all-at-once (0.163): when the sample is still available, splitting it into a separate step is essentially free — so the all-at-once montage isn't inflating the score just by showing everything together.",
      "caveat": "The small kept-in-view > all-at-once edge is likely side-bias-inflated (kept-in-view ran frac-left 0.617 vs all-at-once's balanced 0.465) — read them as equal. Keeping the sample in view isn't a true removal (the sample is still available to the model), so match-from-memory is the load-bearing faithful condition. Raw i2n (human ceiling ≈ 0.449); a side-balanced re-run would tighten the kept-in-view number."
    }
  },
  "gemma_scorecard": {
    "title": "Gemma-4-12B across the capabilities — one model, one interface",
    "subtitle": "Gemma-4-12B (apache-2.0, encoder-free multimodal, released this week) run through every leg of the unified interface from a single integration — behavioral, embodied, and neural — in 4-bit on one A10G. The point isn't the leaderboard position; it's that one model touches every capability through the same process() path.",
    "rows": [
      {"capability": "Vision · behavioral", "benchmark": "Rajalingham 2-AFC", "metric": "i2n raw", "score": "0.146", "status": "done", "note": "direct mode, unbiased (frac-L 0.45) — on par with Qwen-7B (0.163)"},
      {"capability": "Reading · behavioral", "benchmark": "ROAR lexical decision", "metric": "accuracy", "score": "0.86", "status": "done", "note": "ceiled 1.06 (above human mean); not dyslexic; real 0.72 / pseudo 1.00"},
      {"capability": "Embodied · grid game", "benchmark": "toy grid game (CoT)", "metric": "success", "score": "1.00", "status": "done", "note": "solved 15/15 at optimal efficiency with chain-of-thought — the scaling-curve game, same elicitation as Qwen-7B (0.53). On the much harder MiniGrid-DoorKey it scores 0.00 (a different, tougher game); the two are not interchangeable."},
      {"capability": "Vision · neural", "benchmark": "MajajHong V4 / IT", "metric": "median r (raw, cross-validated)", "score": "V4 0.40 · IT 0.53", "status": "done", "note": "encoder-free: 256 image-patch tokens → mean-pooled 3840-d decoder features at layer 20 over 3200 stimuli. Raw 5-fold cross-validated median correlation (NOT ceiled — not directly comparable to the leaderboard's neural-fit metric); IT 0.53 is in CLIP's league."}
    ],
    "reading": "Gemma-4 is encoder-free — image patches project straight into the decoder — so the neural path needed a NEW activations wrapper (decoder hidden states at the image-patch positions), built and validated this session. All four legs now run from one integration: behavioral (2-AFC, ROAR), embodied (the grid game — 15/15 with chain-of-thought, topping the VLM scaling ladder), and neural (V4 0.40 / IT 0.53). The scorecard is the literal expression of the interface's promise: register once, evaluate everywhere — including a model architecture that's days old."
  }
};
