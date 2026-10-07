"""Fitts pointing benchmark on a nodekit site (prototype, Experimental tier).

Each trial: click the home button at the board centre, then click a square
target of width W at distance D. Fitts' law says movement time grows linearly
with the index of difficulty ID = log2(D / W + 1) (Shannon form; Fitts 1954,
MacKenzie 1992). The candidate plays the committed site through
``process(EnvironmentStep)``; its pointer actions are replayed in headless
Chromium and logged by nodekit's own runtime, as for a human.

Score: Pearson r between per-trial movement time (target onset to target
click, page clock) and ID, over trials the candidate completed, floored at 0.
A candidate that finishes fewer than 3 trials scores 0.

Prototype limits: there are no human nodekit traces for this site yet, so the
ceiling is a 1.0 placeholder and the score measures conformity to the law, not
agreement with measured people. ``attrs`` carry the full trace and per-trial
metrics so a human comparison can be added without re-running models.
"""
import json
from pathlib import Path

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore.cursor_traces import read_trials, trajectory_metrics
from brainscore.harnesses.nodekit_browser import NodekitBrowserEnvironment, play_site

SITE_DIR = Path(__file__).parent / 'site'
NODEKIT_VERSION = '0.3.0.dev8'  # commit 3a13ac4; see build_site.py
MIN_TRIALS = 3


def load_design(site_dir=SITE_DIR):
    design = json.loads((Path(site_dir) / 'trials.json').read_text())
    return design, Path(site_dir) / design['entrypoint']


def score_trace(trace, design):
    """Per-trial metrics and the Fitts fit for one trace. Returns (r, info dict)."""
    rows = []
    trials = read_trials(trace)
    targets = {int(t.node_address[0]): t for t in trials if t.node_id == 'target'}
    for i, spec in enumerate(design['trials']):
        row = dict(spec, trial=i, index_of_difficulty=float(np.log2(spec['distance'] / spec['width'] + 1)),
                   completed=False)
        trial = targets.get(i)
        if trial is not None and trial.completed:
            row.update(trajectory_metrics(trial, start=(0, 0), target=(spec['x'], spec['y']),
                                          target_size=(spec['width'], spec['width'])))
            row['completed'] = True
        rows.append(row)
    done = [r for r in rows if r['completed']]
    info = dict(per_trial=rows, completed=len(done), n_trials=len(rows),
                completion_rate=len(done) / len(rows), slope_ms_per_bit=float('nan'),
                intercept_ms=float('nan'), fitts_r=float('nan'))
    if len(done) >= MIN_TRIALS:
        ids = np.array([r['index_of_difficulty'] for r in done])
        mts = np.array([r['response_time'] for r in done])
        if np.ptp(ids) > 0 and np.ptp(mts) > 0:
            info['slope_ms_per_bit'], info['intercept_ms'] = map(float, np.polyfit(ids, mts, 1))
            info['fitts_r'] = float(np.corrcoef(ids, mts)[0, 1])
    r = info['fitts_r']
    return (max(r, 0.0) if np.isfinite(r) else 0.0), info


class NodekitFittsBenchmark(BenchmarkBase):
    """Fitts pointing on nodekit's browser runtime; see the module docstring.

    :param max_actions_per_node: action budget per node (2 nodes per trial)
        before the episode is cut off.
    """

    def __init__(self, site_dir=SITE_DIR, max_actions_per_node: int = 4):
        self.design, self.site_html = load_design(site_dir)
        self.max_steps = max_actions_per_node * 2 * len(self.design['trials'])
        self.required_modalities = set()   # embodied: no perceptual-modality gate
        super().__init__(identifier='Nodekit-fitts-pointing', version=1, parent='embodied',
                         ceiling=Score(1.0), bibtex='')

    def __call__(self, candidate) -> Score:
        with NodekitBrowserEnvironment(
                self.site_html, max_steps=self.max_steps, nodekit_version=NODEKIT_VERSION,
                instruction='Click the grey square in the middle, then click the grey square '
                            'that appears.') as env:
            trace = play_site(candidate, env)
        value, info = score_trace(trace, self.design)
        score = Score(value)
        score.attrs['raw'] = Score(value)
        score.attrs['ceiling'] = float(self.ceiling)
        score.attrs.update({k: v for k, v in info.items() if k != 'per_trial'})
        score.attrs['per_trial'] = info['per_trial']
        score.attrs['trace'] = trace
        return score
