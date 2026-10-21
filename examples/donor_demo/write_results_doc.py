"""Generate experiment/02_results_and_methods.md from the saved result JSONs."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); algo_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
R = json.load(open(D / 'metrics' / 'results_main.json')); M = json.load(open(D / 'MANIFEST.json'))
sel = json.load(open(D / 'demo' / 'selection.json'))
mean = R['target_subject_mean']; boot = R['clip_bootstrap_subject_mean']; null = R['permutation_null']; pu = R['paired_across_units']; per_unit = R['target_per_unit']
CONDS = ['video', 'audio', 'both', 'concat', 'video_pca256', 'audio_pca256', 'both_pca256']
NAMES = {'video': 'Video (V-JEPA v1, 1024-d)', 'audio': 'Soundtrack (Wav2Vec2, 768-d)', 'both': 'Video + soundtrack, banded ridge',
         'concat': 'Video + soundtrack, one penalty', 'video_pca256': 'Video, PCA-256', 'audio_pca256': 'Soundtrack, PCA-256', 'both_pca256': 'Both, PCA-256 each, banded'}
L = []
L.append('# Results and methods: which model inputs predict the brain response?\n')
L.append(f"Generated from `results_main.json` / `MANIFEST.json` (run finished {M['finished_utc']} on {M['instance']}). Raw numbers only in the tables; ceiling-normalized medians are in their own column and are never mixed with raw values.\n")
L.append('## Experiment A: BOLDMoments, V-JEPA v1 video tower vs Wav2Vec2 soundtrack tower\n')
L.append('### Data\n')
L.append('- Stimuli: 1026 three-second clips (BOLDMoments / Memento10k), the registered Brain-Score stimulus set (zip sha1 `dc9c3bf6...`). 19 clips have no audio track; their soundtrack input is a silent placeholder and they are flagged (`stimuli_manifest.csv`). Scores excluding them are in `results_main.json` (`median_r_excluding_silent_clips`).')
L.append('- Responses: Lahner et al. 2024 GLM betas, fsaverage5 (20484 vertices), one value per clip per participant, 10 participants (the assembly axis is named `repetition`; it holds the 10 participants, see assessment). Target for the headline numbers = participant mean; per-participant fits reported separately.')
L.append('- Presentation facts (paper Methods): clips were shown **silently**, 3 s on + 1 s off, fixation task. The soundtrack was never heard. The soundtrack tower therefore measures stimulus information correlated with the picture, not an auditory response.')
L.append('- Vertex sets: visual ROI = split-half reliability (across participants, Spearman-Brown) >= 0.3, 4042 vertices; auditory ROI = Destrieux labels 33/34/35/36/75, 715 vertices; whole cortex = 20484.')
L.append('### Model inputs and features\n')
L.append(f"- Video: V-JEPA v1 ViT-L/16 (`vitl16.pth.tar`, sha1 `{M['model']['video_ckpt_sha1']}`), 16 frames sampled uniformly over the clip, 224x224, ImageNet normalization, fp32; hook on `backbone.blocks.16`; tokens (8 tubelets x 196 patches) averaged over patches by the registered post-hook and then over the 8 tubelets for scoring (1024-d per clip). Token-level activations are kept for the demo clips.")
L.append('- Soundtrack: 16 kHz mono WAV demuxed from each MP4, Wav2Vec2-base `encoder.layers.6` (20 ms frames), averaged over the clip (768-d).')
L.append("- Both towers were run through the registered model `vjepa1-wav2vec2` with `BrainScoreModel.process()`; `start_recording('IT')` routes to the video tower, `start_recording('A1')` to the audio tower. Fresh `RESULTCACHING_HOME`, no cache reuse.")
L.append('### Fitting protocol (identical for every condition)\n')
L.append(f"- Outer CV: {M['protocol']['outer_folds']}-fold over clips, shuffled with seed 0; the fold assignment is saved (`metrics/outer_fold_of_clip.npy`) and shared by all conditions. Clips are discrete 3 s events separated by >= 1 s and randomised across runs in the original experiment, so clip-wise folds do not leak temporally adjacent samples.")
L.append(f"- Penalty: ridge with intercept (centering); alpha chosen per condition, per outer fold and per vertex set by inner {M['protocol']['inner_folds']}-fold validation on the training fold only, grid {list(M['protocol']['alpha_grid'])}; for 'both' one alpha per tower (banded, 36 pairs, joint closed-form solve). PCA-256 variants fit the PCA on the training fold only.")
L.append('- Metric: Pearson r per vertex over held-out clips (all 1026 predictions are out-of-fold), summarised as the median over the vertex set. Ceiling per vertex = sqrt(split-half reliability); normalized score = median over vertices of r / ceiling.')
L.append(f"- Uncertainty: paired bootstrap over clips ({M['protocol']['n_boot']} resamples of the held-out predictions) and paired comparison across the 10 participants (each participant fitted separately with the same folds, 3 inner folds).")
L.append(f"- Null: {M['protocol']['n_perm']} permutations of the clip labels of the training targets inside each fold, with the fitted alphas kept, scored on the real held-out targets.")
L.append('- Timing: model-input time = clip time (0-3 s); brain target = one GLM beta per clip (estimated by Lahner et al. from 1-9 s after onset). Features are averaged over the clip; no HRF convolution is applied to GLM betas. Model time bins are retained in the saved time-resolved assemblies.')
L.append('### Results, participant-mean target\n')
L.append('| condition | visual ROI median r | 95% CI (clips) | normalized (r/ceiling) | auditory ROI r | whole cortex r | alphas chosen (visual, per fold) |')
L.append('|---|---|---|---|---|---|---|')
for c in CONDS:
    v = mean[c]['roi']['visual']; ci = boot['visual'].get(c, {}).get('median_r_ci95', [float('nan')] * 2)
    al = [a['visual'] for a in mean[c]['alphas_per_fold']]
    L.append(f"| {NAMES[c]} | {v['median_r']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {v['median_r_ceiling_normalized']:.3f} | {mean[c]['roi']['auditory']['median_r']:.4f} | {mean[c]['roi']['whole']['median_r']:.4f} | {al} |")
L.append('\nPaired differences across clips (bootstrap of the same held-out predictions):\n')
for roi in ('visual', 'auditory'):
    for k, v in boot[roi].items():
        if '_minus_' in k:
            L.append(f"- {roi}: {k.replace('_minus_', ' - ')}: mean {v['mean']:+.4f}, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], fraction of resamples > 0: {v['frac_positive']:.3f}")
L.append('\n### Results, per participant (10 separate fits)\n')
L.append('| participant | video (visual) | soundtrack (visual) | both (visual) | video (auditory) | soundtrack (auditory) |')
L.append('|---|---|---|---|---|---|')
for u, res in per_unit.items():
    L.append(f"| {u} | {res['video']['roi']['visual']['median_r']:.4f} | {res['audio']['roi']['visual']['median_r']:.4f} | {res['both']['roi']['visual']['median_r']:.4f} | {res['video']['roi']['auditory']['median_r']:.4f} | {res['audio']['roi']['auditory']['median_r']:.4f} |")
L.append('\nPaired across participants:\n')
for roi in ('visual', 'auditory', 'whole'):
    for k, v in pu[roi].items():
        if '_minus_' in k:
            L.append(f"- {roi}: {k.replace('_minus_', ' - ')}: mean {v['mean_diff']:+.4f}, bootstrap 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], Wilcoxon p = {v['wilcoxon_p']:.3g}, {v['n_units_positive']}/{v['n_units']} participants positive")
L.append('\n### Null control\n')
for roi in ('visual', 'auditory'):
    for c in ('video', 'audio', 'both'):
        n = null[roi][c]
        L.append(f"- {roi}, {c}: shuffled-label null mean {n['null_mean']:+.4f} (97.5th percentile {n['null_p975']:+.4f}); observed {n['observed']:.4f}; permutation p = {n['p_value']:.3g}")
L.append('\n### Demo clips\n')
L.append(f"Rule: {sel['rule']}.\n")
L.append('| clip | pattern r (video, visual ROI) | role |')
L.append('|---|---|---|')
disp = sel.get('display', sel['top'])
for c in sel['top']:
    L.append(f"| {c} | {sel['spatial_r'][c]:.3f} | {'shown' if c in disp else 'top-ranked but not displayed (identifiable child)'} |")
L.append('')
L.append(f"Display rule: {sel.get('display_rule', 'top list as is')}.")
L.append(f"| {sel['median']} | {sel['spatial_r'][sel['median']]:.3f} | median clip (not shown) |")
for c in sel['worst']:
    L.append(f"| {c} | {sel['spatial_r'][c]:.3f} | worst (not shown) |")
L.append('\n### Reading\n')
v = mean['video']['roi']['visual']['median_r']; a = mean['audio']['roi']['visual']['median_r']; b = mean['both']['roi']['visual']['median_r']; cc = mean['concat']['roi']['visual']['median_r']
L.append(f"- Video features predict the visual-cortex response well above the null (median r {v:.3f}; normalized {mean['video']['roi']['visual']['median_r_ceiling_normalized']:.2f} of the participant-agreement ceiling).")
L.append(f"- The never-heard soundtrack alone predicts a little ({a:.3f}): sound carries information about what is on screen. Added to video with a per-tower penalty it changes nothing ({b:.3f} vs {v:.3f}; the banded fit pushes the audio penalty to the top of the grid). Concatenating with a single penalty costs accuracy ({cc:.3f}), the expected dilution artefact; it is reported, not shown on the poster.")
L.append('- The PCA-256 matched-budget versions reproduce the pattern, so the video advantage is not a feature-count effect.')
L.append('- Across participants the ordering video > soundtrack holds in every participant; both - video is indistinguishable from zero.')
L.append('- This is the correct answer for a silent experiment, and it is the point of the panel: the toolbox reports a null where there should be one.')
if algo_path and algo_path.exists():
    A = json.load(open(algo_path)); P = A['paired_across_subjects']; ps = P['per_subject_median_r']; pr = A['protocol']
    L.append('\n## Experiment B: Friends with sound (Courtois NeuroMod / Algonauts 2025), four participants\n')
    L.append(f"- Data: Schaefer-1000 parcel BOLD, TR 1.49 s, Friends S1-S6 + Movie10 training runs, participants {', '.join('sub-' + s for s in P['subjects'])}; audiovisual viewing with dialogue.")
    L.append(f"- Features (fixed extractors, pre-existing per-clip caches; video cache sha1 `{A['manifest']['feature_cache_sha1_video_npz'][:12]}...`): {pr['features']['video']}; {pr['features']['audio']}; {pr['features']['text']}. Stimulus window {pr['stimulus_window_TRs']} TRs, HRF delay {pr['hrf_delay_TRs']} TRs, identical for all modalities; first window+delay TRs of each run excluded.")
    L.append(f"- Fitting: outer {pr['outer_folds_over_runs']}-fold CV over whole runs; penalty chosen on {pr['inner_validation_cells_per_outer_fold']} inner run-held-out cells (single-modality grid {list(pr['alpha_grid_single'])}, banded grid {list(pr['alpha_grid_banded'])} per modality); raw Pearson r per parcel, median over 1000 parcels. Null: {pr['null']}. No noise ceiling (no repeated runs in the training set).")
    L.append('\n| condition | ' + ' | '.join('sub-' + s for s in P['subjects']) + ' | mean |')
    L.append('|---|' + '---|' * (len(P['subjects']) + 1))
    for c, vals in ps.items():
        L.append(f"| {c} | " + ' | '.join(f'{x:.4f}' for x in vals) + f" | {sum(vals) / len(vals):.4f} |")
    L.append('\nPaired across participants:\n')
    for k, v in P.items():
        if '_minus_' in k:
            L.append(f"- {k.replace('_minus_', ' - ')}: mean {v['mean_diff']:+.4f} (range {v['min']:+.4f} to {v['max']:+.4f}), {v['n_positive']}/{v['n']} participants positive, Wilcoxon p = {v['wilcoxon_p']:.3g}")
    L.append('\nPer-participant run-bootstrap CIs and timing-shift nulls are in `results_algonauts.json`.')
    s0 = P['subjects'][0]; ns = A['subjects'][s0]['timing_shift_null']
    L.append('Timing-shift null, ' + f'sub-{s0}: ' + '; '.join(f"{c}: null mean {n['null_mean']:+.4f} vs observed {n['observed']:.4f} (p = {n['p_value']:.3g})" for c, n in ns.items()))
L.append('\n## What was not done / caveats\n')
L.append('- No intervention (ablation) condition was run for the poster; the earlier StateChange demonstrations exist on other benchmarks and are not shown here, so no causal claim appears on the slide.')
L.append('- Experiment B uses a still-image video extractor (CLIP per TR) and four participants; it compares information available to an encoding model and does not isolate model families.')
L.append('- Scores in the poster\'s model table come from the registered benchmark protocol (alpha = 1), measured 2026-07-31; they are not comparable with the inner-CV numbers above and are kept in a separate panel.')
L.append('- Vertex/parcel correlations are not independent replications; all uncertainty statements are across clips/runs or across participants.')
OUT.write_text('\n'.join(L))
print('wrote', OUT)
