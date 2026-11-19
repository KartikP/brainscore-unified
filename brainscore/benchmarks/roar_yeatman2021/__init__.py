"""
ROAR (Yeatman 2021) lexical decision — registered variants.

Each leaderboard benchmark declares exactly one input format, so ROAR is
registered as two pinned variants:

- ``Yeatman2021-lexical_decision-image`` — word presented as image
- ``Yeatman2021-lexical_decision-text``  — word presented as text token

Both share the same train/test split, ceiling, and metric — only the
presentation differs.
"""

from functools import partial

from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import Yeatman2021LexicalDecision
    return Yeatman2021LexicalDecision(**configuration)


# Leaderboard variants — each pinned to one input format
benchmark_registry['Yeatman2021-lexical_decision-image'] = partial(
    _load, channels=('vision:image',), identifier='Yeatman2021-lexical_decision-image')
benchmark_registry['Yeatman2021-lexical_decision-text'] = partial(
    _load, channels=('text',), identifier='Yeatman2021-lexical_decision-text')
