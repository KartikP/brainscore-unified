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
            self.manifest['events_sha256'] = hashlib.sha256(event_file.read_bytes()).hexdigest()
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
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported run-record schema version')
        self.codec = PayloadCodec(self.directory)
        self.metadata = self.codec.decode(self.manifest['metadata'])

    def events(self):
        blob = (self.directory / 'events.jsonl').read_bytes()
        expected = self.manifest.get('events_sha256')
        if expected and hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Run event checksum mismatch')
        rows = [json.loads(line) for line in blob.splitlines()]
        if self.manifest.get('event_count', len(rows)) != len(rows):
            raise ValueError('Run event count mismatch')
        for index, row in enumerate(rows):
            if row['sequence'] != index:
                raise ValueError('Run event order mismatch')
            for key in ('args', 'kwargs', 'payload'):
                if key in row:
                    row[key] = self.codec.decode(row[key])
            yield row

    def outputs(self):
        """Replay saved measurements into downstream analyses."""
        for row in self.events():
            if row['kind'] in ('result', 'output'):
                yield row['payload']

    def evaluate(self, metric, target, *, output_index=0):
        """Apply a metric to one saved output. Never invokes model inference."""
        if self.manifest.get("status") != "complete":
            raise ValueError("Measurement replay requires a completed run")
        for index, output in enumerate(self.outputs()):
            if index == output_index:
                return metric(output, target)
        raise IndexError(output_index)
