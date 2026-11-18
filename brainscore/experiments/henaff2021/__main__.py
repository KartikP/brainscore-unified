"""Prepare the archive or run a registered subject through the experiment."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare', help='convert the OSF archive on a remote worker')
    prepare_parser.add_argument('--archive', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    run_parser = commands.add_parser('run', help='view frames, record neural activity, compare with macaques')
    run_parser.add_argument('--prepared', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    run_parser.add_argument('--model', default='clip-vit-b-32')
    run_parser.add_argument('--region', default='V1')
    run_parser.add_argument('--seed', type=int, default=0)
    run_parser.add_argument('--repetitions', type=int, default=1)
    run_parser.add_argument('--provenance', type=Path,
                           help='JSON with checkpoint revision/hashes, device, dtype and settings')
    args = parser.parse_args()
    from .data import load_prepared, prepare, write_json
    if args.command == 'prepare':
        study = prepare(args.archive, args.output)
        print(json.dumps(study.provenance, indent=2))
        return
    import brainscore
    from .experiment import analyze, make_experiment
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    study = load_prepared(args.prepared)
    if args.output.exists():
        raise FileExistsError(f'Use a new run directory: {args.output}')
    subject = brainscore.load_model(args.model)
    experiment = make_experiment(subject, study, args.output, region=args.region,
                                 seed=args.seed, repetitions=args.repetitions, provenance=provenance)
    experiment.validate()
    result = experiment.run()
    write_json(args.output / 'subject_declarations.json', {
        'identifier': subject.identifier, 'in_channels': sorted(subject.in_channels),
        'out_channels': sorted(subject.out_channels),
        'required_channels': sorted(subject.required_channels), 'traits': sorted(subject.traits),
        'region': args.region, 'seed': args.seed})
    for repetition in range(args.repetitions):
        recording = result.neural(args.region, repetition=repetition)
        summary = analyze(recording, study, args.output / f'analysis-{repetition}', provenance=provenance)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
