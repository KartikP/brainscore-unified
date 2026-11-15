"""The plain linear Pereira2018 variants that upstream retired.

Data, metric and ceiling come from the legacy language benchmark. Upstream stopped
registering ``Pereira2018.{experiment}-linear`` because the linear split leaks within a
passage; the release parity check (brainscore.validation.benchmark_parity) still scores
these two identifiers, so they are registered here. A subject reaches them through the
legacy-benchmark facade.
"""

from brainscore_language.benchmarks.pereira2018.benchmark import _Pereira2018Experiment


def Pereira2018_243sentences():
    """Linear-predictivity variant.

    Upstream parameterised the experiment class by metric and stopped registering
    the linear variants. They are restored here because
    ``Pereira2018.{experiment}-linear`` is the identifier the unified-interface
    results are scored against. ``metric='linear_pearsonr'`` with default
    cross-validation reproduces the pre-merge ``_Pereira2018ExperimentLinear``:
    same metric, same CV defaults, same identifier, and with ``atlas='language'``
    the data loader reduces to the old one.
    """
    return _Pereira2018Experiment(experiment='243sentences', metric='linear_pearsonr',
        ceiling_s3_kwargs=dict(
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))


def Pereira2018_384sentences():
    """Linear-predictivity variant — see :func:`Pereira2018_243sentences`."""
    return _Pereira2018Experiment(experiment='384sentences', metric='linear_pearsonr',
        ceiling_s3_kwargs=dict(
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    ))
