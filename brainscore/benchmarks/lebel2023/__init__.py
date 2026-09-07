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

``-languagemask`` reports the mean inside the LanA language network instead of
the median over all cortex, which is the statistic the reference pipeline
publishes. It needs an atlas Brain-Score cannot redistribute, so it is
registered like the others but raises instructions rather than scoring until
that atlas is present: ``python -m brainscore.data lana-atlas``.
"""

from brainscore import benchmark_registry

from .benchmark import (LeBel2023Encoding, LeBel2023EncodingLanguageMask,
                        LeBel2023EncodingWordLevel)

benchmark_registry['LeBel2023-UTS03-encoding'] = (
    lambda: LeBel2023EncodingWordLevel(pooling='lanczos'))
benchmark_registry['LeBel2023-UTS03-encoding-smoke'] = (
    lambda: LeBel2023EncodingWordLevel(
        identifier='LeBel2023-UTS03-encoding-smoke', pooling='lanczos',
        max_targets=2000))

# Retained for comparison: one context window per TR, read at its last token.
benchmark_registry['LeBel2023-UTS03-encoding-contextwindow'] = (
    lambda: LeBel2023Encoding(identifier='LeBel2023-UTS03-encoding-contextwindow'))

# Reports the reference pipeline's statistic. Same fit as the default variant;
# only the summary differs. Requires the LanA atlas (a local, user-supplied
# asset) and says so if it is absent.
benchmark_registry['LeBel2023-UTS03-encoding-languagemask'] = (
    lambda: LeBel2023EncodingLanguageMask())
