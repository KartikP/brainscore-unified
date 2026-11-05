"""Six steps, with both recorders visible. Run remotely from the repository root."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import brainscore
from brainscore.experiments import Experiment, RecordBehavior, RecordNeural
from brainscore.experiments.sun2025 import analyze, make_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    # Check source identity and data availability before loading model weights.
    stimuli = brainscore.load_stimulus_set('Sun2025-second-order-motion', root=args.data)
    reference = brainscore.load_dataset('Sun2025-second-order-motion', root=args.data)
    from brainscore_core import BrainScoreModel
    control = BrainScoreModel(identifier='sun-no-motion-control', supported_modalities={'video'},
                              reply=lambda prompt: '{"direction_deg": 0, "speed_px_per_frame": 0}')
    control_run = Experiment(subject=control, protocol=make_protocol(stimuli),
                             tools=[RecordBehavior()], output_dir=args.output / 'no-motion-control').run()
    _, _, control_summary = analyze(control_run.behavior(measure='text'), stimuli, reference)
    (args.output / 'control-summary.json').write_text(json.dumps(control_summary, indent=2, allow_nan=False) + '\n')

    # 1. Choose a subject. This example uses the native video path of a small VLM.
    from qwen_video import build_subject
    subject, provenance = build_subject()
    provenance['packages'] = {name: version(name) for name in
                              ('torch', 'transformers', 'numpy', 'pandas', 'opencv-python')}

    # 2. Load the registered stimuli and define the motion-report task.
    protocol = make_protocol(stimuli)

    # 3. Attach both instruments. The same presentation produces both outputs.
    recorders = [RecordBehavior(), RecordNeural(['V1'])]

    # 4. Validate and run. TaskProtocol owns session creation and resets.
    experiment = Experiment(subject=subject, protocol=protocol, tools=recorders,
                            output_dir=args.output / 'model', metadata={'execution_provenance': provenance})
    experiment.validate()
    result = experiment.run()

    # 5. Inspect the saved measurements.
    behavior = result.behavior(measure='text')
    neural = result.neural('V1')

    # 6. Load the human data independently and compare the same measured quantities.
    trials, comparison, summary = analyze(behavior, stimuli, reference)
    trials.to_csv(args.output / 'motion-reports.csv', index=False)
    comparison.to_csv(args.output / 'human-comparison.csv', index=False)
    summary['neural_shape'] = list(neural.shape)
    summary['provenance'] = provenance
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
