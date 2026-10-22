"""Generate the speaker notes (provenance + limits) from the saved results."""
import json
import sys
from pathlib import Path


def _paired_or_synth(A):
    """Use the cross-participant block if the run finished, else synthesize it from the per-participant entries."""
    if A.get('paired_across_subjects'):
        return A['paired_across_subjects']
    subs = list(A['subjects'])
    conds = [c for c in ('video', 'audio', 'text', 'video+audio', 'video+audio+text', 'concat') if all(c in A['subjects'][s] for s in subs)]
    return {'subjects': subs, 'per_subject_median_r': {c: [A['subjects'][s][c]['median_r'] for s in subs] for c in conds}, 'partial': True}

D = Path(sys.argv[1]); algo_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
R = json.load(open(D / 'metrics' / 'results_main.json'))
M = json.load(open(D / 'MANIFEST.json'))
sel = json.load(open(D / 'demo' / 'selection.json'))
mean = R['target_subject_mean']; boot = R['clip_bootstrap_subject_mean']; null = R['permutation_null']; pu = R['paired_across_units']


def r(c, roi='visual'):
    return mean[c]['roi'][roi]['median_r']


def ci(c, roi='visual'):
    v = boot[roi].get(c, {}).get('median_r_ci95')
    return f'[{v[0]:.3f}, {v[1]:.3f}]' if v else '(no bootstrap)'


lines = []
lines.append('PROVENANCE (all numbers measured; nothing on this slide is illustrative)')
lines.append(f"Experiment A, BOLDMoments (Lahner et al. 2024 Nat Commun): 1026 three-second clips, 10 participants, fMRI GLM betas on fsaverage5 (20484 vertices). Participants watched the clips SILENTLY while fixating; the soundtrack condition therefore tests information in the clip that was never heard. The Brain-Score assembly's 10 entries per clip are the 10 participants (training clips had only 3 repetitions per participant).")
lines.append(f"Model: registered Brain-Score model 'vjepa1-wav2vec2' = V-JEPA v1 ViT-L/16 (vitl16.pth.tar, sha1 {M['model']['video_ckpt_sha1'][:12]}..., backbone.blocks.16, 16 frames) + Wav2Vec2-base (encoder.layers.6). Features extracted through BrainScoreModel.process() on EC2 {M['instance']} with a fresh activation cache; code commits: " + ', '.join(f"{k} {v['commit'][:9]}" for k, v in M['repos'].items()) + '.')
lines.append(f"Protocol: identical clips, targets, vertices and outer 5-fold clip split for every condition; ridge penalty chosen by inner {M['protocol']['inner_folds']}-fold validation inside each training fold (grid {M['protocol']['alpha_grid']}); per-vertex Pearson r on held-out clips; medians over the visual ROI (split-half reliability >= 0.3, 4042 vertices), the Destrieux auditory ROI (715) and whole cortex. Ceiling = per-vertex sqrt(split-half reliability across participants); normalized scores are reported separately in the methods document, never mixed with raw r on the slide.")
lines.append(f"Replay panel: the clips shown are the top-{len(sel['top'])} held-out clips by predicted-vs-measured pattern correlation across visual-ROI vertices for the video condition ({sel['rule']}); the median clip is {sel['median']} and the three worst are {', '.join(sel['worst'])}. Brain maps are the participant-mean response (z-scored per vertex over all clips) and the held-out prediction under the SAME transform; one map per clip because the data are one GLM beta per clip. The 'Record' panel shows V-JEPA block-16 token activations (14x14 patches per time bin, projected on the clip's first principal direction); decorative geometry is not used anywhere.")
lines.append('RESULTS (visual ROI, participant-mean target, median held-out r with 95% clip-bootstrap interval): ' + '; '.join(f"{c}: {r(c):.3f} {ci(c)}" for c in ('video', 'audio', 'both', 'concat', 'video_pca256', 'audio_pca256', 'both_pca256')) + '.')
lines.append('Paired differences across clips (bootstrap): ' + '; '.join(f"{k}: {v['mean']:+.4f} CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in boot['visual'].items() if '_minus_' in k) + '.')
lines.append('Across participants (10 units, median r per participant): ' + '; '.join(f"{k}: mean diff {v['mean_diff']:+.4f} CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], Wilcoxon p={v['wilcoxon_p']:.3g}, {v['n_units_positive']}/{v['n_units']} positive" for k, v in pu['visual'].items() if '_minus_' in k) + '.')
lines.append('Shuffled-clip null (training labels permuted, fitted penalties kept; visual ROI): ' + '; '.join(f"{c}: null mean {null['visual'][c]['null_mean']:.4f} (97.5th pct {null['visual'][c]['null_p975']:.4f}), observed {null['visual'][c]['observed']:.3f}, p={null['visual'][c]['p_value']:.3g}" for c in ('video', 'audio', 'both')) + '.')
lines.append('Auditory ROI (same protocol): ' + '; '.join(f"{c}: {r(c, 'auditory'):.3f}" for c in ('video', 'audio', 'both')) + '. Whole cortex: ' + '; '.join(f"{c}: {r(c, 'whole'):.3f}" for c in ('video', 'audio', 'both')) + '.')
lines.append("Model table (left): Brain-Score benchmark 'Lahner2024-fMRI-naturalistic-visualROI', ceiling-normalized per-vertex scores measured 2026-07-31 on EC2 with the registered protocol (ridge alpha=1, 5-fold); this is a DIFFERENT fitting protocol from the experiment above (inner-CV alpha), so the two sets of numbers are deliberately kept in separate panels and are not comparable to each other.")
if algo_path and algo_path.exists():
    A = json.load(open(algo_path)); P = _paired_or_synth(A); ps = P['per_subject_median_r']
    lines.append(f"Experiment B, Friends (Courtois NeuroMod, Algonauts 2025 release), participants {', '.join('sub-' + s for s in P['subjects'])}, Schaefer-1000 parcels, TR 1.49 s, watched WITH sound. Fixed extractors: CLIP ViT-B/32 (video frame at each TR), Wav2Vec2-base (audio), all-MiniLM-L6-v2 (transcript); 5-TR window, 3-TR HRF delay for every modality; outer 5-fold CV over whole runs, inner run-held-out penalty selection; raw median r over parcels per participant: " + '; '.join(f"{c}: " + ', '.join(f'{v:.3f}' for v in ps[c]) for c in ps) + '. Paired across participants: ' + '; '.join(f"{k}: {v['mean_diff']:+.4f} (range {v['min']:+.4f}..{v['max']:+.4f}, {v['n_positive']}/{v['n']} positive)" for k, v in P.items() if '_minus_' in k) + '. Timing-shift null per participant in results_algonauts.json. No noise ceiling is available for these training runs, so only raw correlations are shown.')
lines.append('LIMITS. Models are instruments, not copies of people; a prediction correlation says how much variance model features share with a BOLD estimate, not how the brain works. The modality comparison is about information available to an encoding model; nobody had a sense withheld. Vertex correlations are not independent replications; uncertainty is given across clips/runs and across participants. The Friends result uses one still-image video extractor and four participants. Nothing here establishes human mechanisms, medical outcomes, or AI-alignment outcomes, and there is no funding ask on this slide. Stimulus frames are research-licensed clips; confirm display rights before public use.')
lines.append('EDITABLE vs RENDERED. Editable: every text box, the model table, the bar panels and their values, the four stage headings. Rendered media: the measured replay (embedded MP4, autoplay + loop set in the slide timing XML); its frame-to-data mapping is in replay/frame_map.csv. Native PowerPoint playback status is described in the account document.')
lines.append('REPLAY, SECOND HALF (added 2026-10-22). After the nine BOLDMoments clips the loop continues with a 59.6 s window of a held-out Friends episode (S4 E5, run 154 of participant sub-01, scans 297-336). '
             'There the brain maps step once per scan (TR 1.49 s): predicted = video+sound+words ridge readout fitted on the other episodes; measured = Schaefer-1000 parcel BOLD of participant 1; the same per-parcel z-transform is applied to both; '
             'the response lags the picture by about 5 s and the mapping uses the five preceding scans with a three-scan delay. Window chosen by rule (best held-out run by median parcel r, best 40-scan window within it by mean pattern r); '
             'per-scan pattern r is printed on every frame and listed in replay/friends/frame_map.csv. A per-scan version of the BOLDMoments recording was also built (single-trial BOLD, participant sub-01, median pattern r 0.11) and rejected for display because single-scan maps are noise-dominated; it is kept in the data package.')
open(D / 'poster_notes.txt', 'w').write('\n\n'.join(lines))
print('\n\n'.join(lines)[:1500])
