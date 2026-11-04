"""LEACE fitting delegates to concept-erasure; saved maps need no fitter to apply."""
from dataclasses import replace
from importlib.metadata import version
import math
from typing import Any, Mapping, Sequence

import numpy as np

from brainscore_core.events import Selection
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from ._activity_data import _fitting_data, _fitting_provenance
from ._fitted_transform import FittedActivityTransform


class LEACE(FittedActivityTransform):
    """Fit categorical linear concept erasure on explicitly supplied discovery data."""
    identifier = 'leace'
    name = 'leace'

    def __init__(self, target: str | Selection, *, shrinkage: bool = True,
                 constrain_cov_trace: bool = True, svd_tol: float = 0.01,
                 name: str | None = None, conditions: Any = None, repetitions: Any = None) -> None:
        if type(shrinkage) is not bool or type(constrain_cov_trace) is not bool:
            raise ValueError('shrinkage and constrain_cov_trace must be boolean')
        if isinstance(svd_tol, bool) or not isinstance(svd_tol, (int, float)) or not math.isfinite(svd_tol) or svd_tol <= 0:
            raise ValueError('svd_tol must be a positive finite number')
        super().__init__(target, name=name, conditions=conditions, repetitions=repetitions)
        self.shrinkage, self.constrain_cov_trace, self.svd_tol = shrinkage, constrain_cov_trace, float(svd_tol)

    def fit(self, activity: NeuroidAssembly, labels: Sequence[str], *,
            provenance: Mapping[str, Any] | None = None) -> 'LEACE':
        """Fit a fresh tool using float64 features and one-hot categorical labels."""
        try:
            from concept_erasure import LeaceEraser
        except ImportError as error:
            raise ImportError('Install brainscore[interpretability] to fit LEACE') from error
        import torch
        values, units, labels = _fitting_data(activity, labels, self.target)
        classes, encoded = np.unique(labels, return_inverse=True)
        settings = {'shrinkage': self.shrinkage, 'constrain_cov_trace': self.constrain_cov_trace,
                    'svd_tol': self.svd_tol}
        eraser = LeaceEraser.fit(torch.from_numpy(values), torch.from_numpy(np.eye(len(classes))[encoded]),
                                affine=True, **settings)
        fitted = LEACE(replace(self.target, indices=units), name=self.name,
                       conditions=self.conditions, repetitions=self.repetitions, **settings)
        fitted._install({
            'proj_left': eraser.proj_left.detach().cpu().numpy().tolist(),
            'proj_right': eraser.proj_right.detach().cpu().numpy().tolist(),
            'bias': eraser.bias.detach().cpu().numpy().tolist(),
            'classes': classes.tolist(), 'settings': settings,
        }, {**_fitting_provenance(values, units, labels, dict(provenance or {})),
            'concept_erasure_version': version('concept-erasure'), 'torch_version': torch.__version__})
        return fitted

    @classmethod
    def _validate_parameters(cls, parameters: dict, width: int) -> dict[str, np.ndarray]:
        if not isinstance(parameters, dict) or set(parameters) != {
                'proj_left', 'proj_right', 'bias', 'classes', 'settings'}:
            raise ValueError('Invalid LEACE parameters')
        arrays = {key: np.asarray(parameters[key], dtype=np.float64) for key in ('proj_left', 'proj_right', 'bias')}
        left, right, bias = (arrays[key] for key in ('proj_left', 'proj_right', 'bias'))
        classes, settings = parameters['classes'], parameters['settings']
        if (width <= 0 or left.ndim != 2 or left.shape[0] != width or not 0 < left.shape[1] <= width
                or right.shape != (left.shape[1], width) or bias.shape != (width,)
                or any(not np.isfinite(array).all() for array in arrays.values())
                or not isinstance(classes, list) or len(classes) < 2
                or any(not isinstance(label, str) or not label for label in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError('Invalid LEACE parameters or classes')
        if not isinstance(settings, dict) or set(settings) != {'shrinkage', 'constrain_cov_trace', 'svd_tol'}:
            raise ValueError('Invalid LEACE fitting settings')
        if (type(settings['shrinkage']) is not bool or type(settings['constrain_cov_trace']) is not bool
                or isinstance(settings['svd_tol'], bool) or not isinstance(settings['svd_tol'], (int, float))
                or not math.isfinite(settings['svd_tol']) or settings['svd_tol'] <= 0):
            raise ValueError('Invalid LEACE fitting settings')
        return arrays

    @classmethod
    def _constructor_settings(cls, parameters: dict) -> dict:
        return dict(parameters['settings'])

    def _numpy(self, values: np.ndarray, configuration: dict) -> np.ndarray:
        return values - ((values - self._arrays['bias']) @ self._arrays['proj_right'].T) @ self._arrays['proj_left'].T

    def _torch(self, values: Any, parameters: dict, configuration: dict) -> Any:
        return values - ((values - parameters['bias']) @ parameters['proj_right'].T) @ parameters['proj_left'].T
