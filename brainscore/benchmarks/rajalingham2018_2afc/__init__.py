"""Faithful Rajalingham2018 2-AFC — the model makes the actual side-by-side
match-to-sample choice, scored as a subject against the human pool.

The original ``Rajalingham2018-i2n`` reconstructs the binary task from a global
24-way classifier softmax; this variant presents the sample plus two object
choice tokens and scores whichever per-trial choice the model makes (generation,
similarity, readout, or chance) through the identical i1/i2n d-prime pipeline.

Public surface:
  - ``benchmark.score_choices`` / ``benchmark.score_all`` — score a choices DataFrame
  - ``benchmark.load_trials`` / ``benchmark.load_tokens`` / ``benchmark.human_trials_subset``
  - ``montage.compose_montage`` — render one trial (what the model sees)

NOTE: this is intentionally a scoring *library*, not a ``load_benchmark`` target.
The model makes its own per-trial choices (it isn't driven by a single
``benchmark(candidate)`` call), so this subpackage deliberately registers
nothing in ``benchmark_registry`` and is not imported by ``benchmarks/__init__``.
Score via the functions above.
"""
from . import benchmark, montage  # noqa: F401
