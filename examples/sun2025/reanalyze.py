"""Apply the versioned study parser to saved UMI recordings, preserving earlier analysis."""
import argparse
import json
from pathlib import Path

import brainscore
from brainscore.data.sun2025.data_packaging.prepare import checksum
from brainscore.experiments.runner import ExperimentResult
from brainscore.experiments.sun2025 import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stimuli = brainscore.load_stimulus_set('Sun2025-second-order-motion', root=args.data)
    reference = brainscore.load_dataset('Sun2025-second-order-motion', root=args.data)
    result = ExperimentResult(args.run / 'model', None)
    trials, comparison, summary = analyze(result.behavior(measure='text'), stimuli, reference)
    previous = json.loads((args.run / 'summary.json').read_text())
    summary.update(neural_shape=list(result.neural('V1').shape), provenance=previous['provenance'],
                   analysis_method='reanalysis of saved raw replies; no new model inference',
                   previous_summary_sha256=checksum(args.run / 'summary.json'),
                   previous_valid_answers=previous['valid_answers'])
    args.output.mkdir(parents=True, exist_ok=False)
    trials.to_csv(args.output / 'motion-reports.csv', index=False)
    comparison.to_csv(args.output / 'human-comparison.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
