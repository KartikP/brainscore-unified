"""NodeKit tasks as behavior benchmarks (``benchmark.py``); the task runs in ``brainscore.harnesses.nodekit``."""
from functools import partial

from brainscore import benchmark_registry


def _load(identifier: str, task: str, **configuration):
    from .benchmark import NodeKitBenchmark
    return NodeKitBenchmark(identifier, task, **configuration)


benchmark_registry['NodeKit.DriftingGabor'] = partial(_load, 'NodeKit.DriftingGabor', 'drifting-gabor')
benchmark_registry['NodeKit.TripletSimilarity'] = partial(_load, 'NodeKit.TripletSimilarity', 'triplet-similarity')
benchmark_registry['NodeKit.FaceArrangement'] = partial(_load, 'NodeKit.FaceArrangement', 'face-arrangement')
