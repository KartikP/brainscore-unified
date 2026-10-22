#!/usr/bin/env python
"""Append the 2026-10-22 additions (moving-brain replays, lesion protocol) to 02_results_and_methods.md."""
import json, sys
from pathlib import Path
doc, friends_params, tr_json, lesion_summary, out = sys.argv[1:6]
F = json.load(open(friends_params)); T = json.load(open(tr_json)); S = json.load(open(lesion_summary))
L = [open(doc).read().rstrip(), '', '## Addendum 2026-10-22: moving brain maps and the lesion protocol', '']
L += ['### Why the first replay shows one brain map per clip', '',
      'BOLDMoments provides one GLM response amplitude per clip per vertex (window 1 to 9 s after onset, TR 1.75 s), so a comparison can only be made once per clip. Two ways to show a brain map that moves with the picture were built and compared on the same criteria (every frame a real held-out comparison; signal visible to a viewer).', '']
L += ['### Option 1 (selected): Friends, scan-by-scan', '',
      f"Participant sub-01, held-out run {F['run_index']} (episode {F['episode']}), scans {F['window_TRs'][0]}-{F['window_TRs'][1]} ({F['window_episode_time_s'][0]:.1f}-{F['window_episode_time_s'][1]:.1f} s of the episode). Predicted map: video+sound+words ridge readout from Experiment B (held-out, fitted on the other runs); measured: Schaefer-1000 parcel BOLD. Selection rule: {F['selection_rule']}. Display: {F['z_scoring']}; colour range {F['color_range_z']}; parcels drawn on fsaverage5 ({F['parcels']}). {F['hemodynamic_lag_note']}.", '']
r = [v for v in F['per_tr_pattern_r'].values()]
L += [f"Per-scan pattern correlation in the window: median {sorted(r)[len(r)//2]:.2f}, range {min(r):.2f} to {max(r):.2f} (every value is printed on its frame and listed in frame_map.csv). Across all held-out runs of participant 1 the median per-scan pattern r is 0.27 and the median per-parcel temporal r 0.23 (video+sound+words).", '']
L += ['### Option 2 (built, not displayed): BOLDMoments single-trial BOLD, scan-by-scan', '',
      f"Continuous-time encoding model on the ds005165 single-trial time series: V-JEPA v1 block-16 clip features at clip onsets, double-gamma HRF, ridge per vertex, BOLD z-scored per run, leave-one-participant-out with the penalty chosen by inner leave-one-participant-out on the visual ROI (grid {T['protocol']['alpha_grid']}). Pooled held-out median r: visual ROI {T['pooled']['median_r_roi']:.3f}, whole cortex {T['pooled']['median_r_whole']:.3f}; median per-scan pattern r over the ROI {T['pooled']['median_per_tr_spatial_r_roi']:.3f}. Per participant (ROI): " + ', '.join(f"{k} {v['median_r_roi']:.3f}" for k, v in T['per_participant'].items()) + '.', '',
      'This matches the documented single-trial regime for this dataset (0.07 to 0.09). Rendered as a replay (replay/boldmoments_per_scan/), the predicted map is barely visible and the measured map is dominated by scan-to-scan noise, so it was not placed on the poster. It is kept so the comparison can be inspected.', '']
P = S['protocol']
L += ['### Lesion protocol (slide 2)', '',
      f"Subjects: Qwen2.5-VL-Instruct at three sizes ({', '.join(S['models'])}); agents per size: greedy decoding plus temperatures {[t for t in P['temps'] if t > 0]} x seeds {P['seeds']}. Task: Yeatman2021 lexical decision, image modality, generation path (the model answers 'real' or 'pseudo'), 100 held-out images, 400 training images for the localizer. Recorded and lesioned sites: five MLP outputs evenly spaced between {P['depth_range'][0]:.0%} and {P['depth_range'][1]:.0%} of depth; K = {P['frac_units_per_layer']:.0%} of the hidden width per layer. Sessions: before, targeted lesion (K most made-up-word-selective units by Cohen d), random lesion (K random units, seed 0), opposite lesion (K most real-word-selective units), restored (reset). Every session re-presents the same 100 images and records answers and layer activity.", '']
L += ['| model | agents | before | targeted | random | opposite | restored | targeted > random (agents) | reliable (paired trial bootstrap) | restored identical |', '|---|---|---|---|---|---|---|---|---|---|']
for m in sorted(S['models'], key=lambda m: S['models'][m]['hidden']):
    mr = S['models'][m]; pop = mr['population']; n = pop['baseline']['n_agents']
    def mp(sess): 
        v = [a['sessions'][sess]['accuracy_pseudo'] for a in mr['agents'].values()]; return f"{sum(v)/len(v):.2f}"
    L.append(f"| {m} | {n} | {mp('baseline')} | {mp('lesion_pseudo_selective')} | {mp('control_random')} | {mp('lesion_real_selective')} | {mp('restored')} | {pop['targeted_minus_random_deficit']['n_agents_targeted_worse_than_random']}/{n} | {pop['targeted_minus_random_deficit']['n_agents_reliable']}/{n} | {pop['restored_identical_all_agents']} |")
L += ['', 'Values are made-up-word accuracy, mean over agents (all-word and real-word accuracies, per-agent values, trial-bootstrap intervals and per-trial tables are in `lesion_summary.json`, `results_lesion.json` and `trials.csv`).', '']
if 'dose' in S:
    L += ['### Lesion size (greedy agents)', '', '| model | fraction silenced | before | targeted | random | opposite |', '|---|---|---|---|---|---|']
    for m, rows in S['dose'].items():
        for fr, d in rows.items():
            L.append(f"| {m} | {fr} | {d['baseline']:.2f} | {d['lesion_pseudo_selective']:.2f} | {d['control_random']:.2f} | {d['lesion_real_selective']:.2f} |")
    L.append('')
Path(out).write_text('\n'.join(L) + '\n'); print('wrote', out)
