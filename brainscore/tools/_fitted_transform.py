"""Artifact and lifecycle support for fitted, reversible activity transforms."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from brainscore.experiments.tools import Ablate
from brainscore_core.events import Selection
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from ._activity_data import _activity, _canonical, _checksum


class FittedActivityTransform(Ablate):
    """Share artifacts and attachment; concrete methods supply their numerical maps."""
    operation = 'transform'
    identifier: str
    application_keys: frozenset[str] = frozenset()

    def __init__(self, target: str | Selection, *, name: str | None = None,
                 conditions: Any = None, repetitions: Any = None) -> None:
        target = Selection(target) if isinstance(target, str) else deepcopy(target)
        if not isinstance(target, Selection) or not target.layer:
            raise ValueError('A transform needs a named target or Selection')
        super().__init__([target], name=name, conditions=conditions, repetitions=repetitions)
        self._parameters: dict | None = None
        self._provenance: dict = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._target_signature: str | None = None

    @property
    def target(self) -> Selection:
        return deepcopy(self.targets[0])

    @property
    def fitted(self) -> bool:
        return self._parameters is not None

    @property
    def provenance(self) -> dict:
        return deepcopy(self._provenance)

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError('Call fit on discovery activity or load a fitted artifact first')
        if _canonical(asdict(self.target)) != self._target_signature:
            raise ValueError('Fitted target changed; fit a new tool for another population')

    @classmethod
    def _validate_parameters(cls, parameters: dict, width: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def _constructor_settings(cls, parameters: dict) -> dict:
        return {}

    def _install(self, parameters: dict, provenance: dict) -> None:
        units = self.target.indices
        if (not units or any(type(i) is not int or i < 0 for i in units)
                or len(set(units)) != len(units)):
            raise ValueError('Fitted transforms require distinct original integer unit addresses')
        if not isinstance(provenance, dict):
            raise ValueError('Invalid transform provenance')
        _canonical([parameters, provenance, asdict(self.target)])
        arrays = self._validate_parameters(parameters, len(units))
        self._arrays = {key: np.array(array, dtype=np.float64, copy=True) for key, array in arrays.items()}
        for array in self._arrays.values():
            array.setflags(write=False)
        self._parameters, self._provenance = deepcopy(parameters), deepcopy(provenance)
        self._target_signature = _canonical(asdict(self.target))

    def _payload(self) -> dict:
        self._require_fitted()
        payload = {'format': 'brainscore.activity-transform', 'schema_version': 1,
                   'method': self.identifier, 'target': asdict(self.target),
                   'parameters': deepcopy(self._parameters), 'provenance': self.provenance}
        configuration = self._application_configuration()
        if configuration:
            payload['application_defaults'] = configuration
        return payload

    def _application_configuration(self) -> dict:
        return {}

    def save(self, path: str | PathLike) -> None:
        """Save fitted parameters and provenance without pickle or overwriting."""
        payload = self._payload()
        encoded = _canonical({'payload': payload, 'sha256': _checksum(payload)}) + '\n'
        with Path(path).open('x') as handle:
            handle.write(encoded)

    @classmethod
    def load(cls, path: str | PathLike, *, name: str | None = None,
             conditions: Any = None, repetitions: Any = None, **configuration: Any) -> 'FittedActivityTransform':
        """Restore the selected method only; artifact contents never import code."""
        saved = json.loads(Path(path).read_text())
        payload = saved['payload']
        if saved['sha256'] != _checksum(payload):
            raise ValueError('Transform artifact checksum mismatch')
        if (payload['format'] != 'brainscore.activity-transform' or payload['schema_version'] != 1
                or payload['method'] != cls.identifier):
            raise ValueError('Unsupported transform artifact method, format or version')
        target = Selection(**payload['target'])
        if target.indices is not None:
            target.indices = tuple(target.indices)
        cls._validate_parameters(payload['parameters'], len(target.indices or ()))
        settings = cls._constructor_settings(payload['parameters'])
        defaults = payload.get('application_defaults', {})
        if not isinstance(defaults, dict) or set(defaults) - cls.application_keys:
            raise ValueError('Invalid transform application defaults')
        configuration = {**defaults, **configuration}
        if set(settings) & set(configuration):
            raise TypeError('Fitting settings are fixed by the artifact; fit a new tool to change them')
        tool = cls(target, name=name, conditions=conditions, repetitions=repetitions, **settings, **configuration)
        tool._install(payload['parameters'], payload['provenance'])
        return tool

    def _numpy(self, values: np.ndarray, configuration: dict) -> np.ndarray:
        raise NotImplementedError

    def _torch(self, values: Any, parameters: dict, configuration: dict) -> Any:
        raise NotImplementedError

    def transform(self, activity: NeuroidAssembly) -> NeuroidAssembly:
        """Apply saved parameters to an assembly, preserving coordinates and dtype."""
        self._require_fitted()
        values, _ = _activity(activity, self.target, label=self.identifier)
        if activity.dtype.kind != 'f':
            raise ValueError('Activity transforms require floating-point activity')
        configuration = deepcopy(self._application_configuration())
        transformed = self._numpy(values, configuration)
        if transformed.shape != values.shape or not np.isfinite(transformed).all():
            raise ValueError('An activity transform produced invalid values or shape')
        cast = transformed.astype(activity.dtype)
        if not np.isfinite(cast).all():
            raise ValueError('Transformed values overflow the activity dtype')
        result = NeuroidAssembly(activity.copy(deep=True))
        result.data = cast
        application = {'backend': 'numpy', 'numpy_version': np.__version__,
                       'calculation_dtype': 'float64', 'dtype': str(activity.dtype)}
        if configuration:
            application['configuration'] = configuration
        result.attrs = {**deepcopy(activity.attrs), 'transform': self.identifier,
                        'tool_sha256': _checksum(self._payload()), 'fitting_provenance': self.provenance,
                        'application_provenance': application}
        return result

    def describe(self) -> dict:
        return {**super().describe(), 'operation': self.identifier,
                'artifact_sha256': _checksum(self._payload()), 'provenance': self.provenance,
                'application_dtype': 'activity dtype', 'positions': 'each position independently',
                **self._application_configuration()}

    def validate(self, experiment: Any) -> None:
        self._require_fitted()
        super().validate(experiment)

    @contextmanager
    def attach(self, context: Any) -> Iterator[None]:
        self._require_fitted()
        source, targets = self.name, deepcopy(self.targets)
        configuration = deepcopy(self._application_configuration())
        description = self.describe()
        conditions, repetitions = self.conditions, self.repetitions
        context.publish('tool_artifact', self._payload(), source=source)
        if ((conditions is not None and context.condition not in conditions)
                or (repetitions is not None and context.repetition_id not in repetitions)):
            yield
            return
        import torch
        cache = {}

        def apply(tensor):
            key = (tensor.device, tensor.dtype)
            if key not in cache:
                cache[key] = {name: torch.tensor(array, device=tensor.device, dtype=tensor.dtype)
                              for name, array in self._arrays.items()}
            result = self._torch(tensor, cache[key], configuration)
            application = {
                'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'device': str(tensor.device),
                'torch_version': str(torch.__version__),
            }
            if configuration:
                application['configuration'] = configuration
            context.publish('intervention_application', application, source=source)
            return result

        try:
            with context.experiment.instrumentation.transform(targets, apply=apply):
                context.publish('intervention_start', description, source=source)
                try:
                    yield
                finally:
                    context.publish('intervention_end', description, source=source)
        finally:
            cache.clear()
