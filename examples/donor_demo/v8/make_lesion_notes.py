#!/usr/bin/env python
"""Speaker notes for slide 2 from lesion_summary.json (+ results_lesion.json protocol block)."""
import json, sys
S = json.load(open(sys.argv[1])); out = sys.argv[2]
P = S['protocol']; L = []
L.append('SLIDE 2 PROVENANCE. Lesion protocol run on EC2 (g6.12xlarge, 4x L4) on 2026-10-22 with the Brain-Score unified interface: '
         'BrainScoreModel(generation_fn, state_change_fn=build_pytorch_ablation_fn) ; process(StateChange(kind="ablation", target=Selection(layer, indices), perturbation=Perturbation("zero"))) ; reset(). '
         f'Benchmark Yeatman2021-lexical_decision-image (400 training / 100 held-out word images, human ceiling 0.811, pass mark {0.65}). '
         f'Models: {", ".join(S["models"])} (Qwen2.5-VL-Instruct family, bfloat16). Decoding: temperatures {P["temps"]}, seeds {P["seeds"]} (temperature 0 is greedy, one agent). '
         f'Recorded and lesioned layers: five MLP outputs evenly spaced between {int(P["depth_range"][0]*100)}% and {int(P["depth_range"][1]*100)}% of depth; '
         f'K = {P["frac_units_per_layer"]:.0%} of the hidden width per layer. Localizer: Cohen d (real minus made-up) on mean MLP activity over all 400 training images, greedy single forward pass.')
for m, mr in S['models'].items():
    L.append(f'{m.upper()}: hidden {mr["hidden"]}, K={mr["K"]} per layer, layers {", ".join(p.split(".")[-2] for p in mr["layers"])}.')
    for ag, ar in mr['agents'].items():
        ss = ar['sessions']
        row = ' | '.join(f"{k.replace('lesion_pseudo_selective','targeted').replace('control_random','random').replace('lesion_real_selective','opposite')}: all {ss[k]['accuracy']:.2f} real {ss[k]['accuracy_real']:.2f} made-up {ss[k]['accuracy_pseudo']:.2f}" for k in ['baseline','lesion_pseudo_selective','control_random','lesion_real_selective','restored'])
        ci = ss['lesion_pseudo_selective']['deficit_ci95']
        L.append(f'  agent {ag}: {row}. Targeted deficit (all words) {ss["lesion_pseudo_selective"]["deficit_vs_baseline"]:+.2f} [95% trial bootstrap {ci[0]:+.2f}, {ci[1]:+.2f}]; '
                 f'random {ss["control_random"]["deficit_vs_baseline"]:+.2f}; answers changed by reset: {ss["restored"]["answers_changed"]}.')
    pop = mr['population']
    L.append(f'  population: targeted worse than random in {pop["targeted_minus_random_deficit"]["n_agents_targeted_worse_than_random"]} of {pop["baseline"]["n_agents"]} agents, beyond the 95% paired trial-bootstrap interval in {pop["targeted_minus_random_deficit"]["n_agents_reliable"]}; restored answers identical for all agents: {pop["restored_identical_all_agents"]}.')
L.append('LIMITS. A "lesion" here is zeroing a unit\'s output; it is a statement about which computations the answer depends on in this model, not about human brains. '
         'Made-up versus real is a two-choice test, so a model that answers "real" to everything scores 0.5 overall and 0 on made-up words. '
         'Temperature changes the dice, not the subject: the agents of one size share weights, so they are repeats of one subject, not independent subjects; sizes are different models trained by the same group. '
         'Random control uses one random draw per model (seed 0). No causal claim beyond this model family and this task.')
open(out, 'w').write('\n\n'.join(L)); print('wrote', out)
