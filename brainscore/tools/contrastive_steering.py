"""Learn explicit mean-difference directions and apply signed activation additions."""
from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from brainscore_core.events import Selection
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from ._activity_data import _fitting_data, _fitting_provenance
from ._fitted_transform import FittedActivityTransform


class ContrastiveSteering(FittedActivityTransform):
    """Add strength times a saved positive-minus-negative group-mean direction."""
    identifier = 'contrastive-steering'
    name = 'contrastive_steering'
    application_keys = frozenset({'strength'})

    def __init__(self, target: str | Selection, *, strength: float = 1.,
                 name: str | None = None, conditions: Any = None, repetitions: Any = None) -> None:
        super().__init__(target, name=name, conditions=conditions, repetitions=repetitions)
        self.strength = strength

    @property
    def strength(self) -> float:
        """Application dose; changes take effect on the next attachment."""
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError('strength must be a finite number')
        self._strength = float(value)

    def fit(self, activity: NeuroidAssembly, labels: Sequence[str], *, positive: str,
            negative: str, normalize: bool = False,
            provenance: Mapping[str, Any] | None = None) -> 'ContrastiveSteering':
        """Fit a fresh direction on two explicit groups; never choose token pooling."""
        if (not isinstance(positive, str) or not positive or not isinstance(negative, str)
                or not negative or positive == negative or type(normalize) is not bool):
            raise ValueError('Choose distinct nonempty contrast labels and a boolean normalize setting')
        values, units, labels = _fitting_data(activity, labels, self.target)
        pos, neg = labels == positive, labels == negative
        if not pos.any() or not neg.any() or not np.all(pos | neg):
            raise ValueError('Every label must belong to the requested contrast, with both groups present')
        try:
            with np.errstate(over='raise', invalid='raise'):
                direction = values[pos].mean(axis=0) - values[neg].mean(axis=0)
        except FloatingPointError as error:
            raise ValueError('Contrastive direction overflowed; check the supplied activity scale') from error
        if not np.isfinite(direction).all():
            raise ValueError('Contrastive direction must be finite')
        if normalize:
            scale = np.max(np.abs(direction))
            if scale == 0:
                raise ValueError('Cannot normalize a zero contrastive direction')
            direction = direction / scale
            direction = direction / np.linalg.norm(direction)
        fitted = ContrastiveSteering(replace(self.target, indices=units), strength=self.strength,
                                     name=self.name, conditions=self.conditions, repetitions=self.repetitions)
        fitted._install({'direction': direction.tolist(), 'positive': positive, 'negative': negative,
                         'normalized': normalize, 'counts': {'positive': int(pos.sum()), 'negative': int(neg.sum())}},
                        _fitting_provenance(values, units, labels, dict(provenance or {})))
        return fitted

    @classmethod
    def _validate_parameters(cls, parameters: dict, width: int) -> dict[str, np.ndarray]:
        if not isinstance(parameters, dict) or set(parameters) != {
                'direction', 'positive', 'negative', 'normalized', 'counts'}:
            raise ValueError('Invalid contrastive steering parameters')
        direction = np.asarray(parameters['direction'], dtype=np.float64)
        positive, negative, normalized, counts = (parameters[key] for key in (
            'positive', 'negative', 'normalized', 'counts'))
        if (width <= 0 or direction.shape != (width,) or not np.isfinite(direction).all()
                or not isinstance(positive, str) or not positive or not isinstance(negative, str)
                or not negative or positive == negative or type(normalized) is not bool
                or not isinstance(counts, dict) or set(counts) != {'positive', 'negative'}
                or any(type(count) is not int or count <= 0 for count in counts.values())):
            raise ValueError('Invalid contrastive steering parameters')
        if normalized and not np.isclose(np.linalg.norm(direction), 1., rtol=1e-10, atol=1e-12):
            raise ValueError('Invalid normalized steering parameters')
        return {'direction': direction}

    def _application_configuration(self) -> dict:
        return {'strength': self.strength}

    def _numpy(self, values: np.ndarray, configuration: dict) -> np.ndarray:
        if configuration['strength'] == 0:
            return values.copy()
        return values + configuration['strength'] * self._arrays['direction']

    def _torch(self, values: Any, parameters: dict, configuration: dict) -> Any:
        if configuration['strength'] == 0:
            return values.clone()
        return values + configuration['strength'] * parameters['direction']
