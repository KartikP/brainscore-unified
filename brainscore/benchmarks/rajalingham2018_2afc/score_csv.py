"""Score a choices CSV (image_id, sample_obj, dist_obj, choice) against the human
pool via the same i1/i2n pipeline as every other model. Used to score the output
of standalone choosers run in a different env (e.g. gemma_chooser.py). Run in bsu.

    python -m brainscore.benchmarks.rajalingham2018_2afc.score_csv \\
        --csv /tmp/raj2afc/gemma12b_direct/gemma_choices.csv \\
        --out /tmp/raj2afc/gemma12b_direct/result.json
"""
import argparse
import json
import os

import pandas as pd

from . import benchmark as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    acc = float((df['choice'] == df['sample_obj']).mean())
    scores = B.score_all(df[['image_id', 'sample_obj', 'dist_obj', 'choice']], metrics=('i1', 'i2n'))
    summ = {}
    sj = os.path.join(os.path.dirname(args.csv), 'summary.json')
    if os.path.exists(sj):
        summ = json.load(open(sj))
    summ.update({'accuracy': acc, 'scores': scores, 'n_trials': int(len(df))})
    json.dump(summ, open(args.out, 'w'), indent=2, default=str)
    print(json.dumps({'model': summ.get('model'), 'mode': summ.get('mode'),
                      'accuracy': round(acc, 3), 'i2n': scores['i2n'], 'i1': scores['i1']},
                     indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
