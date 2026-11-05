"""A laboratory procedure, recording equipment, human reference and scoring rule."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from brainscore_core import Score
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.contract import Subject
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore.experiments import Experiment, RecordBehavior, TaskProtocol
from brainscore.experiments.sun2025 import make_protocol

from . import IDENTIFIER
from .metric import motion_vector_error


def _validate_reference(stimuli, reference, *, full_study=False):
    """Check the study design and matched human observations before any model runs."""
    if stimuli.empty or stimuli.stimulus_id.duplicated().any():
        raise ValueError('Use nonempty stimuli with distinct stimulus identities')
    groups = stimuli.groupby('base_id')
    if stimuli.drop_duplicates('base_id').duplicated(['scene', 'modulation']).any():
        raise ValueError('Each scene/modulation pair must identify exactly one group')
    for _, rows in groups:
        if (sorted(rows.flip) != [1, 2, 3, 4] or rows.scene.nunique() != 1
                or rows.modulation.nunique() != 1):
            raise ValueError('Each scene/modulation group must contain all four distinct flips')
    reference_ids = list(reference['stimulus_id'].values.astype(str))
    if (len(reference_ids) != len(set(reference_ids))
            or set(reference_ids) != set(stimuli.base_id)):
        raise ValueError('Human reference identities must match the stimulus groups exactly')
    if (set(reference.dims) != {'presentation', 'participant', 'axis'}
            or list(reference['axis'].values) != ['x', 'y']
            or list(reference['participant'].values) != [1, 2, 3]
            or not np.isfinite(reference.values).all()):
        raise ValueError('Use finite x/y reference vectors from the three study participants')
    by_id = stimuli.drop_duplicates('base_id').set_index('base_id')
    for coordinate in ('scene', 'modulation'):
        if list(reference[coordinate].values) != list(by_id.loc[reference_ids, coordinate]):
            raise ValueError(f'Human reference {coordinate} coordinates do not match stimuli')
    if full_study:
        expected = {(scene, modulation, flip) for scene in range(1, 41)
                    for modulation in range(1, 8) for flip in range(1, 5)}
        actual = set(zip(stimuli.scene, stimuli.modulation, stimuli.flip))
        if (len(stimuli) != 1120 or actual != expected
                or reference.attrs.get('provenance', {}).get('complete') is not True):
            raise ValueError('The registered Sun benchmark requires the complete 1120-trial study; '
                             'compose Sun2025MotionBenchmark(protocol, reference) for a pilot')


def _custom_identifier(protocol, reference):
    ordered = reference.isel(presentation=np.argsort(reference['stimulus_id'].values))
    identity = {'protocol': protocol.identifier, 'instruction': protocol.task.instruction,
                'trial_order': list(protocol.stimuli.stimulus_id.astype(str)),
                'reference': reference.attrs.get('identifier'),
                'reference_ids': list(ordered['stimulus_id'].values.astype(str)),
                'reference_values': ordered.transpose('presentation', 'participant', 'axis').values.tolist()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return f'{IDENTIFIER}-custom-{digest}'


class Sun2025MotionBenchmark(BenchmarkBase):
    """Reuse a study protocol; attach recorders, human observations and a metric.

    The default instrument is RecordBehavior. Supply RecordBehavior together
    with RecordNeural to retain activity from those same presentations. The
    behavioral score never requires a neural recording or biological region map.
    """
    uses_session = True
    leaderboard_eligible = False

    def __init__(self, protocol: TaskProtocol, reference: BehavioralAssembly, *,
                 recorders=None, output_dir: str | Path | None = None,
                 identifier: str | None = None, metadata: dict | None = None):
        # Procedure: the existing task protocol defines trials, sessions and resets.
        if len(protocol.conditions) != 1 or len(protocol.repetitions) != 1:
            raise ValueError('This benchmark scores one condition and one repetition')
        _validate_reference(protocol.stimuli, reference, full_study=identifier == IDENTIFIER)
        self.protocol = protocol

        # Equipment: request measurements through the ordinary recording tools.
        self.recorders = tuple(recorders) if recorders is not None else (RecordBehavior(),)
        requests = set().union(*(recorder.output_channels for recorder in self.recorders))
        protocol.accept_requests(requests)
        if 'behavior:text' not in protocol.output_channels:
            raise ValueError('Attach RecordBehavior to collect the motion reports')
        self.required_input_channels = protocol.input_channels
        self.requested_output_channels = protocol.output_channels
        self.required_traits = protocol.required_traits

        # Comparison: measured human behavior and a fixed, explicit scoring rule.
        self.reference = reference
        self.metric = motion_vector_error
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.metadata = dict(metadata or {})
        ceiling = Score(float('nan'), attrs={'status': 'not_estimated'})
        super().__init__(identifier=identifier or _custom_identifier(protocol, reference),
                         version=1, parent='behavioral', ceiling=ceiling,
                         bibtex='@article{sun2025motion, title={Machine learning modelling for '
                         'multi-order human visual motion processing}, author={Sun and others}, '
                         'journal={Nature Machine Intelligence}, year={2025}, '
                         'doi={10.1038/s42256-025-01068-w}, '
                         'url={https://doi.org/10.1038/s42256-025-01068-w}}')

    def __call__(self, subject: Subject) -> Score:
        """Run the procedure with its equipment, then compare the measurements."""
        check_channel_compatibility(subject, self)
        if self.output_dir is not None:
            return self._run(subject, self.output_dir)
        with tempfile.TemporaryDirectory(prefix='brainscore-sun2025-') as temporary:
            score = self._run(subject, Path(temporary) / 'experiment')
            score.attrs.update(experiment_directory=None, records_retained=False)
            return score

    def _run(self, subject: Subject, directory: Path) -> Score:
        experiment = Experiment(subject=subject, protocol=self.protocol,
                                tools=self.recorders, output_dir=directory,
                                metadata={**self.metadata, 'benchmark': self.identifier,
                                          'version': self.version})
        result = experiment.run()
        measurements = result.behavior(condition=self.protocol.conditions[0],
                                       repetition=self.protocol.repetitions[0], measure='text')
        score = self.score_measurements(measurements)
        score.attrs.update(experiment_directory=str(result.directory), records_retained=True,
                           experiment_provenance=result.manifest)
        return score

    def score_measurements(self, measurements: BehavioralAssembly) -> Score:
        """Compare already collected reports without showing the stimuli again."""
        score = self.metric(measurements, self.protocol.stimuli, self.reference)
        if 'summary' in score.attrs:
            score.attrs['summary']['full_study'] = self.identifier == IDENTIFIER
        score.attrs.update(benchmark_identifier=self.identifier, benchmark_version=self.version,
                           protocol_identifier=self.protocol.identifier,
                           reference_identifier=self.reference.attrs.get('identifier'),
                           reference_provenance=self.reference.attrs.get('provenance', {}),
                           measurement_provenance=measurements.attrs,
                           execution_provenance=self.metadata.get('execution_provenance', {}),
                           study_doi='10.1038/s42256-025-01068-w',
                           full_study=self.identifier == IDENTIFIER,
                           leaderboard_eligible=False)
        return score


def load_benchmark(*, root: str | Path | None = None, recorders=None,
                   output_dir: str | Path | None = None,
                   metadata: dict | None = None) -> Sun2025MotionBenchmark:
    """Assemble the registered full-study benchmark from the data plugin."""
    from brainscore import load_dataset, load_stimulus_set
    stimuli = load_stimulus_set('Sun2025-second-order-motion', root=root)
    reference = load_dataset('Sun2025-second-order-motion', root=root)
    _validate_reference(stimuli, reference, full_study=True)
    protocol = make_protocol(stimuli, seed=0)
    return Sun2025MotionBenchmark(protocol, reference, recorders=recorders,
                                  output_dir=output_dir, identifier=IDENTIFIER, metadata=metadata)
