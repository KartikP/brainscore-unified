"""
ROAR (Yeatman 2021) lexical decision — registered variants.

Each leaderboard benchmark declares exactly one input format, so ROAR is
registered as two pinned variants:

- ``Yeatman2021-lexical_decision-image`` — word presented as image
- ``Yeatman2021-lexical_decision-text``  — word presented as text token

Both share the same train/test split, ceiling, and metric — only the
presentation differs.
"""

from brainscore import benchmark_registry
from .benchmark import Yeatman2021LexicalDecision

# Leaderboard variants — each pinned to one input format
benchmark_registry['Yeatman2021-lexical_decision-image'] = (
    lambda: Yeatman2021LexicalDecision(channels=('vision:image',)))
benchmark_registry['Yeatman2021-lexical_decision-text'] = (
    lambda: Yeatman2021LexicalDecision(channels=('text',)))
