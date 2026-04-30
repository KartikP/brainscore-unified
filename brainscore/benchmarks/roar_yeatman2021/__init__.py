"""
ROAR (Yeatman 2021) lexical decision — registered variants.

Per the April 30, 2026 design decision, each leaderboard benchmark
declares exactly one input format. ROAR is split into two pinned
variants:

- ``Yeatman2021-lexical_decision-image`` — word presented as image
- ``Yeatman2021-lexical_decision-text``  — word presented as text token

Both share the same train/test split, ceiling, and metric — only the
presentation differs.

The legacy ``Yeatman2021-lexical_decision`` identifier is kept as a
DeprecationWarning alias defaulting to the image variant for back-compat
with existing scoring scripts.
"""

import warnings

from brainscore import benchmark_registry
from .benchmark import Yeatman2021LexicalDecision

# Leaderboard variants — each pinned to one input format
benchmark_registry['Yeatman2021-lexical_decision-image'] = (
    lambda: Yeatman2021LexicalDecision(modality='vision'))
benchmark_registry['Yeatman2021-lexical_decision-text'] = (
    lambda: Yeatman2021LexicalDecision(modality='text'))


# Back-compat alias — defaults to image variant (the original presentation).
def _legacy_alias():
    warnings.warn(
        "Benchmark identifier 'Yeatman2021-lexical_decision' is deprecated. "
        "Use 'Yeatman2021-lexical_decision-image' or "
        "'Yeatman2021-lexical_decision-text' explicitly. "
        "This alias defaults to the image variant.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Yeatman2021LexicalDecision(modality='vision')


benchmark_registry['Yeatman2021-lexical_decision'] = _legacy_alias
