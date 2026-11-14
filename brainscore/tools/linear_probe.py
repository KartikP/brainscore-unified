"""Fitted linear classification probes, usable offline or as experiment tools."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from ._activity_data import _activity, _canonical, _checksum, _fitting_data

from brainscore.experiments.runner import Tool
from brainscore_core.events import Selection
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly,
    walk_coords,
)


class LinearProbe(Tool):
    """A standardized logistic readout that never changes model activity.

    fit returns a new fitted tool. Fitting requires (presentation, neuroid)
    activity with original unit_index coordinates. Prediction supports leading
    position dimensions without silently pooling them. Hooked predictions use
    tensor positions, not inferred stimulus IDs or token identities.
    """

    def __init__(self, target: str | Selection, *, name: str = 'linear_probe',
                 when: str = 'after', classifier_c: float = 1.0) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Probe name must be nonempty')
        if when not in ('before', 'after'):
            raise ValueError('when must be before or after')
        if (isinstance(classifier_c, bool) or not isinstance(classifier_c, (int, float))
                or not math.isfinite(classifier_c) or classifier_c <= 0):
            raise ValueError('classifier_c must be a positive finite inverse penalty C')
        target = Selection(target) if isinstance(target, str) else deepcopy(target)
        if not target.layer:
            raise ValueError('A probe needs a nonempty target')
        self.target = target
        self.name, self.when = name, when
        self.classifier_c = float(classifier_c)
        self.phase = 'before_intervention' if when == 'before' else 'after_intervention'
        self._parameters: dict | None = None

    @property
    def fitted(self) -> bool:
        return self._parameters is not None

    @property
    def provenance(self) -> dict:
        return {} if not self.fitted else deepcopy(self._parameters['provenance'])

    def fit(self, activity: NeuroidAssembly, labels: Sequence[str], *,
            provenance: Mapping[str, Any] | None = None) -> 'LinearProbe':
        """Fit on explicit discovery examples and return an independent tool."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import sklearn

        values, units, labels = _fitting_data(activity, labels, self.target)
        scaler = StandardScaler().fit(values)
        classifier = LogisticRegression(C=self.classifier_c, max_iter=1000)
        classifier.fit(scaler.transform(values), labels)
        if np.any(classifier.n_iter_ >= classifier.max_iter):
            raise ValueError('Probe did not converge; adjust fitting data or regularization')
        fitted = LinearProbe(replace(self.target, indices=units), name=self.name,
                             when=self.when, classifier_c=self.classifier_c)
        fitted._parameters = {
            'classes': classifier.classes_.tolist(), 'weights': classifier.coef_.tolist(),
            'intercept': classifier.intercept_.tolist(), 'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'provenance': {**deepcopy(dict(provenance or {})),
                           'n_fitting_examples': len(values), 'sklearn_version': sklearn.__version__,
                           'fitting_sha256': hashlib.sha256(
                               values.tobytes() + _canonical([list(units), labels.tolist()]).encode()
                           ).hexdigest()},
        }
        _canonical(fitted._parameters)
        return fitted

    def _payload(self) -> dict:
        if not self.fitted:
            raise RuntimeError('Call fit on discovery activity or load a fitted artifact first')
        return {'format': 'brainscore.linear-probe', 'schema_version': 1,
                'target': {'layer': self.target.layer, 'indices': list(self.target.indices),
                           'axis': self.target.axis, 'expected_shape': self.target.expected_shape,
                           'metadata': deepcopy(self.target.metadata)},
                'classifier_c': self.classifier_c, **deepcopy(self._parameters)}

    def predict(self, activity: NeuroidAssembly) -> BehavioralAssembly:
        """Decode supplied activity without refitting or changing it."""
        from scipy.special import expit, softmax
        parameters = self._payload()
        values, _ = _activity(activity, self.target)
        values = (values - np.asarray(parameters['mean'])) / np.asarray(parameters['scale'])
        logits = values @ np.asarray(parameters['weights']).T + np.asarray(parameters['intercept'])
        if len(parameters['classes']) == 2:
            positive = expit(logits[..., 0])
            probabilities = np.stack((1 - positive, positive), axis=-1)
        else:
            probabilities = softmax(logits, axis=-1)
        coords = {name: (dims, np.asarray(values).copy()) for name, dims, values in walk_coords(activity)
                  if 'neuroid' not in dims}
        return BehavioralAssembly(probabilities, dims=(*activity.dims[:-1], 'choice'),
                                  coords={**coords, 'choice': parameters['classes']},
                                  attrs={'method': 'linear-probe', 'probe_sha256': _checksum(parameters),
                                         'provenance': deepcopy(parameters['provenance'])})

    def save(self, path: str | PathLike) -> None:
        """Save a checksummed, versioned JSON artifact without pickle or overwrite."""
        payload = self._payload()
        text = _canonical({'payload': payload, 'sha256': _checksum(payload)}) + '\n'
        with Path(path).open('x') as handle:
            handle.write(text)

    @classmethod
    def load(cls, path: str | PathLike, *, name: str = 'linear_probe',
             when: str = 'after') -> 'LinearProbe':
        """Load and validate a fitted artifact; dependencies are not deserialized."""
        saved = json.loads(Path(path).read_text())
        payload = saved['payload']
        if saved['sha256'] != _checksum(payload):
            raise ValueError('Probe artifact checksum mismatch')
        if payload['format'] != 'brainscore.linear-probe' or payload['schema_version'] != 1:
            raise ValueError('Unsupported probe artifact format or version')
        target = dict(payload['target'])
        target['indices'] = tuple(target['indices'])
        if target['expected_shape'] is not None:
            target['expected_shape'] = tuple(target['expected_shape'])
        probe = cls(Selection(**target), name=name, when=when,
                    classifier_c=payload['classifier_c'])
        count, classes = len(probe.target.indices), payload['classes']
        rows = 1 if len(classes) == 2 else len(classes)
        if (not count or len(classes) < 2 or len(set(classes)) != len(classes)
                or any(not isinstance(label, str) or not label for label in classes)
                or len(set(probe.target.indices)) != count
                or any(type(i) is not int or i < 0 for i in probe.target.indices)):
            raise ValueError('Invalid probe classes or unit addresses')
        for key, shape in [('weights', (rows, count)), ('intercept', (rows,)),
                           ('mean', (count,)), ('scale', (count,))]:
            array = np.asarray(payload[key], dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f'Invalid probe parameter: {key}')
        if np.any(np.asarray(payload['scale']) <= 0) or not isinstance(payload['provenance'], dict):
            raise ValueError('Invalid probe scale or provenance')
        probe._parameters = {key: deepcopy(payload[key]) for key in (
            'classes', 'weights', 'intercept', 'mean', 'scale', 'provenance')}
        return probe

    def describe(self) -> dict:
        payload = self._payload()
        return {'method': 'linear-probe', 'target': payload['target'], 'when': self.when,
                'artifact_sha256': _checksum(payload), 'provenance': self.provenance}

    def validate(self, experiment: Any) -> None:
        self._payload()
        if experiment.instrumentation is None:
            raise ValueError('LinearProbe requires an instrumentation provider')
        experiment.instrumentation.validate('record', [self.target])

    @contextmanager
    def attach(self, context: Any) -> Iterator[None]:
        payload = self._payload()
        context.publish('probe_artifact', payload, source=self.name)
        captures = 0

        def receive(target: str, value: dict) -> None:
            nonlocal captures
            array = np.asarray(value['array'])
            axis = -1 if self.target.axis is None else self.target.axis
            array = np.moveaxis(array, axis, -1)
            if array.ndim == 1:
                array = array[None, :]
            dims = ('presentation', *(f'position_{i}' for i in range(array.ndim - 2)), 'neuroid')
            assembly = NeuroidAssembly(array, dims=dims, coords={
                'unit_index': ('neuroid', list(self.target.indices)),
                'layer': ('neuroid', [self.target.layer] * len(self.target.indices)),
            })
            context.publish('probe', self.predict(assembly), source=self.name)
            captures += 1

        with context.experiment.instrumentation.record([self.target], receive):
            yield
            if not captures:
                raise RuntimeError(
                    f'Probe {self.name!r} observed no activity at {self.target.layer!r}; '
                    'run a real forward pass rather than using cached-only responses')
