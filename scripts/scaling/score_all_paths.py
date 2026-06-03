"""Score every model through every output path it is capable of, on ROAR.

A feature model (CLIP) can only do the READOUT path; an instruction-tuned VLM
(Qwen, BLIP-2) can do BOTH readout and generation; GPT-2 (text) can do both on
the text modality. We force each path via TaskContext.prefer_path and record the
score per (model, path) so the behavioral plot can colour points by path.

Run on EC2 (GPU). Writes JSON to --out.
"""
import argparse
import json
import os
import time

import numpy as np


# (model_id, modality, [paths to try])
PLAN = [
    ('chance-baseline', 'vision', ['readout']),
    ('random-vit-b-32', 'vision', ['readout']),
    ('clip-vit-b-32', 'vision', ['readout']),
    ('qwen2.5-vl-3b', 'vision', ['readout', 'generation']),
    ('blip2-opt-2.7b', 'vision', ['readout', 'generation']),
    ('gpt2', 'text', ['readout', 'generation']),
]


def score_path(model_id, modality, path, benchmark_image, benchmark_text):
    import brainscore
    from brainscore_core.model_interface import TaskContext
    bench = benchmark_text if modality == 'text' else benchmark_image
    model = brainscore.load_model(model_id)
    tc = TaskContext(
        task_type='probabilities',
        fitting_stimuli=bench._train_stimuli,
        label_set=['real', 'pseudo'],
        instruction='Is this a real word or a pseudo word?',
        prefer_path=path,
    )
    model.start_task(tc)
    preds = model.process(bench._test_stimuli)
    choices = list(preds['choice'].values)
    pred_idx = preds.values.argmax(axis=1)
    predicted = [choices[i] for i in pred_idx]
    stim_ids = list(preds['stimulus_id'].values)
    truth_map = dict(zip(bench._test_stimuli['stimulus_id'].values,
                         bench._test_stimuli['image_label'].values))
    truth = [truth_map[s] for s in stim_ids]
    return float(np.mean([p == t for p, t in zip(predicted, truth)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/all_paths.json')
    args = ap.parse_args()
    import brainscore
    t0 = time.time()
    bench_img = brainscore.load_benchmark('Yeatman2021-lexical_decision-image')
    bench_txt = brainscore.load_benchmark('Yeatman2021-lexical_decision-text')

    results = {'chance': 0.5, 'rows': []}
    for model_id, modality, paths in PLAN:
        for path in paths:
            try:
                acc = score_path(model_id, modality, path, bench_img, bench_txt)
                results['rows'].append({'model': model_id, 'path': path, 'acc': acc})
                print(f'[{time.time()-t0:6.0f}s] {model_id:18s} {path:10s} -> {acc:.3f}', flush=True)
            except Exception as e:
                results['rows'].append({'model': model_id, 'path': path, 'acc': None, 'error': str(e)})
                print(f'[{time.time()-t0:6.0f}s] {model_id:18s} {path:10s} FAILED: {e}', flush=True)
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)
    print('written', args.out)


if __name__ == '__main__':
    main()
