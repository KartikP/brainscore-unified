"""LeBel 2023 whole-cortex encoding — registered variants.

The default reads one feature per spoken word and resamples those onto the fMRI
grid with a windowed-sinc filter. Roughly six words fall inside each 2 s sample,
and collapsing them to the last one discards the rest: measured here that costs
about 20% of the score, and the reference pipeline reports the same ordering
(last < sum ~ average < Lanczos).

The earlier per-TR context-window path is kept as ``-contextwindow`` so the
comparison stays reproducible rather than only described.

``-smoke`` scores a 2000-vertex random subset. It exists so the pipeline can be
exercised in a couple of minutes; its number is not comparable to a full run.
"""

from brainscore import benchmark_registry

from .benchmark import LeBel2023Encoding, LeBel2023EncodingWordLevel

benchmark_registry['LeBel2023-UTS03-encoding'] = (
    lambda: LeBel2023EncodingWordLevel(pooling='lanczos'))
benchmark_registry['LeBel2023-UTS03-encoding-smoke'] = (
    lambda: LeBel2023EncodingWordLevel(
        identifier='LeBel2023-UTS03-encoding-smoke', pooling='lanczos',
        max_targets=2000))

# Retained for comparison: one context window per TR, read at its last token.
benchmark_registry['LeBel2023-UTS03-encoding-contextwindow'] = (
    lambda: LeBel2023Encoding(identifier='LeBel2023-UTS03-encoding-contextwindow'))
