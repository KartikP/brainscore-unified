"""Versioned experiment records with verified array artifacts and offline replay.

No pickle or dynamic class imports. Unknown payload types fail explicitly.
Callers provide model/checkpoint, protocol, environment and tool provenance in
metadata. A record preserves observations; it does not promise deterministic
re-execution of a model or environment.
"""

from dataclasses import fields, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

SCHEMA_VERSION = 1


def _event_types():
    from brainscore_core import events
    from brainscore_core.contract import TaskContext
    from brainscore_core.streaming import StreamEvent
    types = [getattr(events, name) for name in (
        'CameraFrame', 'Proprioception', 'EnvironmentStep', 'EnvironmentResponse',
        'Message', 'StateChange', 'Perturbation', 'PerturbationApplied', 'Selection')]
    return {t.__name__: t for t in [*types, TaskContext, StreamEvent]}


def _stamp(handle):
    # ctime changes on any write, link change or utime call, and cannot be set by the writer.
    status = os.fstat(handle.fileno())
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns,
            status.st_ctime_ns)


class PayloadCodec:
    def __init__(self, directory):
        self.directory = Path(directory)

    def encode(self, value):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {'$float': str(value)}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {'$object_array': self.encode(list(value.flat)), 'shape': list(value.shape)}
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            blob = buffer.getvalue()
            digest = hashlib.sha256(blob).hexdigest()
            assets = self.directory / 'arrays'
            assets.mkdir(exist_ok=True)
            path = assets / f'{digest}.npy'
            try:
                with path.open('xb') as handle:
                    handle.write(blob)
            except FileExistsError:
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Corrupt array artifact: {path}')
            return {'$array': digest}
        if isinstance(value, Path):
            blob = value.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            assets = self.directory / 'files'
            assets.mkdir(exist_ok=True)
            destination = assets / digest
            if destination.exists():
                if destination.read_bytes() != blob:
                    raise ValueError('File artifact checksum mismatch')
            else:
                destination.write_bytes(blob)
            return {'$file': digest, 'name': value.name}
        if isinstance(value, tuple):
            return {'$tuple': [self.encode(v) for v in value]}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        if isinstance(value, dict):
            return {'$mapping': [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if is_dataclass(value) and type(value).__name__ in _event_types():
            if type(value) is not _event_types()[type(value).__name__]:
                raise TypeError('Unregistered event class')
            return {'$event': type(value).__name__, 'fields': {
                f.name: self.encode(getattr(value, f.name)) for f in fields(value)}}
        # Imports are local so a record containing basic events stays lightweight.
        import pandas as pd
        import xarray as xr
        if isinstance(value, pd.DataFrame):
            from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
            result = {'$frame': 'StimulusSet' if isinstance(value, StimulusSet) else 'DataFrame',
                      'columns': self.encode(list(value.columns)),
                      'index': self.encode(list(value.index)),
                      'data': self.encode(value.to_numpy()), 'attrs': self.encode(value.attrs),
                      'dtypes': [str(t) for t in value.dtypes]}
            if isinstance(value, StimulusSet):
                result['identifier'] = getattr(value, 'identifier', None)
                result['stimulus_paths'] = self.encode({key: Path(path) for key, path in
                    getattr(value, 'stimulus_paths', {}).items()})
            return result
        if isinstance(value, xr.DataArray):
            if type(value).__name__ not in ('DataArray', 'Score', 'DataAssembly',
                                            'NeuroidAssembly', 'BehavioralAssembly'):
                raise TypeError(f'Unsupported assembly type: {type(value).__name__}')
            return {'$assembly': type(value).__name__, 'dims': list(value.dims),
                    'data': self.encode(value.values), 'attrs': self.encode(value.attrs),
                    'name': self.encode(value.name),
                    'multi_indexes': {dim: list(index.names) for dim, index in value.indexes.items()
                                      if dim in value.dims and isinstance(index, pd.MultiIndex)},
                    'coords': {k: {'dims': list(v.dims), 'data': self.encode(v.values)}
                               for k, v in value.coords.items()}}
        raise TypeError(f'Unsupported record payload: {type(value).__module__}.{type(value).__name__}. '
                        'Use a StreamEvent with dict/list/array payloads or a supported assembly.')

    def decode(self, value):
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        if '$float' in value:
            if value['$float'] not in ('nan', 'inf', '-inf'):
                raise ValueError('Invalid nonfinite float encoding')
            return float(value['$float'])
        if '$array' in value:
            digest = value['$array']
            if not isinstance(digest, str) or len(digest) != 64 or any(
                    c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid array artifact identity')
            blob = (self.directory / 'arrays' / f'{digest}.npy').read_bytes()
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError('Array artifact checksum mismatch')
            return np.load(io.BytesIO(blob), allow_pickle=False)
        if '$object_array' in value:
            items = self.decode(value['$object_array'])
            result = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):
                result[i] = item
            return result.reshape(value['shape'])
        if '$file' in value:
            digest = value['$file']
            if not isinstance(digest, str) or len(digest) != 64 or any(
                    c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid file artifact identity')
            path = self.directory / 'files' / digest
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('File artifact checksum mismatch')
            return path
        if '$tuple' in value:
            return tuple(self.decode(v) for v in value['$tuple'])
        if '$mapping' in value:
            return {self.decode(k): self.decode(v) for k, v in value['$mapping']}
        if '$event' in value:
            cls = _event_types().get(value['$event'])
            if cls is None:
                raise ValueError('Unknown event schema')
            return cls(**{k: self.decode(v) for k, v in value['fields'].items()})
        if '$frame' in value:
            import pandas as pd
            from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
            classes = {'DataFrame': pd.DataFrame, 'StimulusSet': StimulusSet}
            result = classes[value['$frame']](self.decode(value['data']),
                        columns=self.decode(value['columns']), index=self.decode(value['index']))
            for column, dtype in zip(result.columns, value.get('dtypes', [])):
                result[column] = result[column].astype(dtype)
            result.attrs.update(self.decode(value['attrs']))
            if isinstance(result, StimulusSet):
                result.identifier = value['identifier']
                result.stimulus_paths = self.decode(value['stimulus_paths'])
            return result
        if '$assembly' in value:
            import xarray as xr
            from brainscore_core.supported_data_standards.brainio import assemblies
            from brainscore_core.metrics import Score
            classes = {'DataArray': xr.DataArray, 'Score': Score}
            for name in ('DataAssembly', 'NeuroidAssembly', 'BehavioralAssembly'):
                classes[name] = getattr(assemblies, name)
            if value['$assembly'] not in classes:
                raise ValueError(f"Unsupported assembly type: {value['$assembly']}")
            import pandas as pd
            coords = {k: (v['dims'], self.decode(v['data'])) for k, v in value['coords'].items()}
            for dim, names in value.get('multi_indexes', {}).items():
                entries = list(coords[dim][1])
                for name in names:
                    coords.pop(name, None)
                coords[dim] = pd.MultiIndex.from_tuples(entries, names=names)
            return classes[value['$assembly']](self.decode(value['data']), dims=value['dims'],
                coords=coords,
                attrs=self.decode(value['attrs']), name=self.decode(value['name']))
        raise ValueError('Unknown payload encoding')


class RunRecorder:
    """An observer and session sink writing to a new directory.

    Each record is flushed when appended. Inputs are serialized before inference
    so model mutations cannot rewrite history. Incomplete/error runs remain
    explicitly marked and readable for diagnosis.
    """

    def __init__(self, directory, *, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.codec = PayloadCodec(self.directory)
        self.manifest = {'schema_version': SCHEMA_VERSION, 'run_id': uuid.uuid4().hex,
                         'metadata': self.codec.encode(metadata), 'status': 'open'}
        self._sequence = 0
        self._closed = False
        self._failed = False
        (self.directory / 'events.jsonl').touch()
        self._write_manifest()

    def _write_manifest(self):
        temporary = self.directory / 'manifest.tmp'
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.directory / 'manifest.json')

    def _append(self, record):
        if self._closed:
            raise RuntimeError('RunRecorder is closed')
        record['sequence'] = self._sequence
        with (self.directory / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        self._sequence += 1

    def on_start(self, call):
        self._append({'kind': 'call', 'event_id': call.event_id, 'method': call.method,
                      'started_ns': call.started_ns, 'args': self.codec.encode(call.args),
                      'kwargs': self.codec.encode(call.kwargs)})

    def on_result(self, call, result):
        self._append({'kind': 'result', 'event_id': call.event_id,
                      'duration_s': call.duration_s, 'payload': self.codec.encode(result)})

    def on_error(self, call, error):
        self._failed = True
        self._append({'kind': 'error', 'event_id': call.event_id,
                      'duration_s': call.duration_s,
                      'error': {'type': type(error).__name__, 'message': str(error)}})

    def record(self, direction, event):
        if direction not in ('input', 'output'):
            raise ValueError('direction must be input or output')
        self._append({'kind': direction, 'payload': self.codec.encode(event)})

    def close(self, *, failed=False):
        if not self._closed:
            self.manifest['status'] = 'failed' if failed or self._failed else 'complete'
            self.manifest['event_count'] = self._sequence
            event_file = self.directory / 'events.jsonl'
            if not event_file.exists():
                event_file.write_text('')
            with event_file.open('rb') as handle:
                self.manifest['events_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
            self._write_manifest()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(failed=exc_type is not None)


class RunRecord:
    """Read and verify a stored run without loading its model."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.experiment_manifest = None
        experiment = self.directory / 'experiment.json'
        if experiment.is_file():
            self.experiment_manifest = json.loads(experiment.read_text())
            if self.experiment_manifest.get('schema_version') != 1:
                raise ValueError('Unsupported experiment schema version')
            self.directory = self.directory / 'inputs_outputs'
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported run-record schema version')
        self.codec = PayloadCodec(self.directory)
        self.metadata = self.codec.decode(self.manifest['metadata'])

    def events(self):
        """Yield events after the whole log passes checksum, JSON, count and order checks.

        Two passes over one open file: memory is bounded by the largest event line,
        not by log length. Callers that collect events into a list are not bounded.
        """
        expected = self.manifest.get('events_sha256')
        with (self.directory / 'events.jsonl').open('rb') as handle:
            # Concurrency limit: no lock or snapshot copy. For a closed record, the file's
            # identity, size, mtime and ctime are taken before validation and rechecked after
            # it and before each yield, so any write the filesystem timestamps fails closed.
            # This does not defend against a writer who can forge ctime (clock or root
            # access) or whose write lands within the filesystem's timestamp resolution of
            # the previous change. Open records are read as an unchecked diagnostic prefix.
            initial = _stamp(handle) if expected else None

            def check_unchanged():
                if initial and _stamp(handle) != initial:
                    raise ValueError('Run event log changed while reading')

            digest, size, count, parse_error, order_error = hashlib.sha256(), 0, 0, None, None
            for chunk in iter(handle.readline, b''):
                digest.update(chunk)
                size += len(chunk)
                for line in chunk.splitlines():
                    if parse_error is None:
                        try:
                            row = json.loads(line)
                        except ValueError as error:  # reported after the checksum, as before
                            parse_error = error
                        else:
                            try:
                                if order_error is None and row['sequence'] != count:
                                    order_error = ValueError('Run event order mismatch')
                            except (KeyError, TypeError) as error:
                                order_error = error
                    count += 1
            verified = digest.hexdigest()
            if expected and verified != expected:
                raise ValueError('Run event checksum mismatch')
            if parse_error is not None:
                raise parse_error
            if self.manifest.get('event_count', count) != count:
                raise ValueError('Run event count mismatch')
            if order_error is not None:
                raise order_error
            check_unchanged()
            handle.seek(0)
            digest, index = hashlib.sha256(), 0
            while size:
                chunk = handle.readline(size)
                if not chunk:
                    break
                size -= len(chunk)
                digest.update(chunk)
                for line in chunk.splitlines():
                    row = json.loads(line)
                    if row['sequence'] != index:
                        raise ValueError('Run event log changed while reading')
                    index += 1
                    for key in ('args', 'kwargs', 'payload'):
                        if key in row:
                            row[key] = self.codec.decode(row[key])
                    if self.experiment_manifest is not None:
                        # Experiment records carry lifecycle and condition metadata in an envelope.
                        # Standalone RunRecord callers retain their original representation.
                        envelope = row['payload']
                        if not isinstance(envelope, dict) or 'kind' not in envelope:
                            raise ValueError('Invalid experiment event envelope')
                        row = envelope
                    check_unchanged()
                    yield row
            if index != count or digest.hexdigest() != verified:
                raise ValueError('Run event log changed while reading')

    def outputs(self):
        """Replay saved measurements into downstream analyses."""
        for row in self.events():
            if row['kind'] in ('result', 'output'):
                yield row['payload']

    def evaluate(self, metric, target, *, output_index=0):
        """Apply a metric to one saved output. Never invokes model inference."""
        if (self.manifest.get("status") != "complete"
                or (self.experiment_manifest is not None
                    and self.experiment_manifest.get("status") != "complete")):
            raise ValueError("Measurement replay requires a completed run")
        for index, output in enumerate(self.outputs()):
            if index == output_index:
                return metric(output, target)
        raise IndexError(output_index)
