#!/usr/bin/env python
"""Summarise the lesion protocol: per model x agent x session accuracies, deficits with trial
bootstrap, agreement across agents (population), restoration check, and development figures."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SESS = ['baseline', 'lesion_pseudo_selective', 'control_random', 'lesion_real_selective', 'restored']
NICE = {'baseline': 'before', 'lesion_pseudo_selective': 'targeted\nlesion', 'control_random': 'random\nlesion',
        'lesion_real_selective': 'opposite\nlesion', 'restored': 'restored'}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n_boot', type=int, default=2000); ap.add_argument('--dose_dir', default=None); a = ap.parse_args()
    D = Path(a.data); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    res = json.load(open(D / 'results_lesion.json')); tr = pd.read_csv(D / 'trials.csv')
    rng = np.random.default_rng(0)
    summary = {'protocol': res['protocol'], 'models': {}}
    for model, mr in res['models'].items():
        ms = {'K': mr['K'], 'hidden': mr['hidden'], 'layers': mr['layers'], 'agents': {}, 'population': {}}
        t = tr[tr.model == model]
        for agent, ar in mr['agents'].items():
            ta = t[t.agent == agent]
            base = ta[ta.session == 'baseline'].set_index('stimulus_id')
            rows = {}
            for s in SESS:
                ts = ta[ta.session == s].set_index('stimulus_id').loc[base.index]
                d = {k: float(v) for k, v in ar['sessions'][s].items() if v is not None and k.startswith('accuracy')}
                if s != 'baseline':   # paired per-trial deficit with trial bootstrap
                    diff = (base.correct.values - ts.correct.values).astype(float)
                    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(a.n_boot)]
                    d['deficit_vs_baseline'] = float(diff.mean()); d['deficit_ci95'] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                    pm = base.truth.values == 'pseudo'
                    d['deficit_pseudo'] = float(diff[pm].mean()); d['deficit_real'] = float(diff[~pm].mean())
                    d['answers_changed'] = int((base.answer.values != ts.answer.values).sum())
                rows[s] = d
            rows['restored_identical_answers'] = bool(rows['restored'].get('answers_changed', 1) == 0)
            # paired per-trial targeted-vs-random: positive = targeted lesion cost more than the random one
            tt = ta[ta.session == 'lesion_pseudo_selective'].set_index('stimulus_id').loc[base.index]
            tc = ta[ta.session == 'control_random'].set_index('stimulus_id').loc[base.index]
            dtr = (tc.correct.values - tt.correct.values).astype(float)
            bt = [dtr[rng.integers(0, len(dtr), len(dtr))].mean() for _ in range(a.n_boot)]
            rows['targeted_minus_random'] = {'mean': float(dtr.mean()), 'ci95': [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                                             'reliable': bool(np.percentile(bt, 2.5) > 0)}
            ms['agents'][agent] = {'temperature': ar['temperature'], 'seed': ar['seed'], 'sessions': rows, 'targeted_minus_random': rows.pop('targeted_minus_random')}
        # population: every agent of this model
        pop = {}
        for s in SESS:
            accs = [ms['agents'][ag]['sessions'][s]['accuracy'] for ag in ms['agents']]
            pop[s] = {'mean': float(np.mean(accs)), 'min': float(np.min(accs)), 'max': float(np.max(accs)), 'n_agents': len(accs)}
        defs = [ms['agents'][ag]['sessions']['lesion_pseudo_selective']['deficit_vs_baseline'] for ag in ms['agents']]
        ctrl = [ms['agents'][ag]['sessions']['control_random']['deficit_vs_baseline'] for ag in ms['agents']]
        pop['targeted_minus_random_deficit'] = {'per_agent': [float(x - y) for x, y in zip(defs, ctrl)],
                                                'n_agents_targeted_worse_than_random': int(sum(x > y for x, y in zip(defs, ctrl))),
                                                'n_agents_reliable': int(sum(ms['agents'][ag]['targeted_minus_random']['reliable'] for ag in ms['agents']))}
        pop['restored_identical_all_agents'] = bool(all(ms['agents'][ag]['sessions']['restored_identical_answers'] for ag in ms['agents']))
        ms['population'] = pop
        summary['models'][model] = ms
    if a.dose_dir and Path(a.dose_dir).exists():
        dose = {}
        for d in sorted(Path(a.dose_dir).glob('*_f*')):
            f = d / 'results_lesion.json'
            if not f.exists(): continue
            R = json.load(open(f)); frac = R['protocol']['frac_units_per_layer']
            for m, mr in R['models'].items():
                ag = mr['agents'].get('T0_s0')
                if ag: dose.setdefault(m, {})[f'{frac:.2f}'] = {s_: ag['sessions'][s_]['accuracy_pseudo'] for s_ in SESS} | {'K': mr['K']}
        for m in summary['models']:   # the main run is the 0.25 point
            ag = summary['models'][m]['agents'].get('T0_s0')
            if ag: dose.setdefault(m, {})['0.25'] = {s_: ag['sessions'][s_]['accuracy_pseudo'] for s_ in SESS} | {'K': summary['models'][m]['K']}
        summary['dose'] = {m: dict(sorted(v.items())) for m, v in dose.items()}
    json.dump(summary, open(O / 'lesion_summary.json', 'w'), indent=1)
    # ---- dev figure: accuracy per session, one panel per model, one line per agent ----
    models = list(summary['models']); fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for ag, ar in summary['models'][m]['agents'].items():
            y = [ar['sessions'][s]['accuracy'] for s in SESS]
            ax.plot(range(5), y, '-o', color='#2a78d6' if ar['temperature'] == 0 else '#9ab9e3', lw=2 if ar['temperature'] == 0 else 1, ms=4, alpha=0.9)
            yp = [ar['sessions'][s]['accuracy_pseudo'] for s in SESS]
            ax.plot(range(5), yp, '--', color='#eb6834', lw=1, alpha=0.6)
        ax.axhline(0.65, color='#999', lw=1, ls=':'); ax.set_xticks(range(5)); ax.set_xticklabels([NICE[s] for s in SESS], fontsize=8)
        ax.set_title(m); ax.set_ylim(0, 1.02); ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('accuracy on 100 held-out words'); fig.tight_layout(); fig.savefig(O / 'dev_accuracy_by_session.png', dpi=150)
    # ---- dev figure: recorded activity raster, greedy agent, deepest recorded layer, baseline vs targeted lesion ----
    for m in models:
        L = summary['models'][m]['layers'][-1]
        try:
            b = np.load(D / 'activations' / f'{m}__T0_s0__baseline.npz'); l = np.load(D / 'activations' / f'{m}__T0_s0__lesion_pseudo_selective.npz')
        except FileNotFoundError: continue
        A = b[f'acts__{L}'].astype(np.float32); B = l[f'acts__{L}'].astype(np.float32)
        units = json.load(open(D / 'localizer' / f'{m}_units.json'))['units'][L]['lesion_pseudo_selective']
        order = np.concatenate([units, np.setdiff1d(np.arange(A.shape[1]), units)])
        v = np.percentile(np.abs(A), 98)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, M_, ttl in zip(axes, (A, B), ('before', 'targeted lesion')):
            ax.imshow(M_[:, order].T, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest'); ax.set_title(f'{m}: {ttl}'); ax.set_xlabel('trial (word image)'); ax.set_ylabel('unit (silenced first)')
        fig.tight_layout(); fig.savefig(O / f'dev_raster_{m}.png', dpi=150); plt.close(fig)
        # clean version for the slide: two panels, minimal chrome, paper background
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), facecolor='#f6f5f1')
        for ax, M_, ttl in zip(axes, (A, B), ('before', 'targeted lesion')):
            ax.imshow(M_[:, order].T, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
            ax.set_title(ttl, fontsize=11, color='#141414', loc='left'); ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(len(units) - 0.5, color='#141414', lw=0.8); ax.set_facecolor('#f6f5f1')
            for sp in ax.spines.values(): sp.set_visible(False)
        axes[0].set_ylabel('units  (silenced set above the line)', fontsize=9, color='#6e6e69'); axes[0].set_xlabel('100 word images', fontsize=9, color='#6e6e69'); axes[1].set_xlabel('100 word images', fontsize=9, color='#6e6e69')
        fig.tight_layout(); fig.savefig(O / f'raster_clean_{m}.png', dpi=200, facecolor='#f6f5f1'); plt.close(fig)
    print(json.dumps({m: summary['models'][m]['population'] for m in models}, indent=1)[:3000])

if __name__ == '__main__':
    main()
