"""LeBel 2023 whole-cortex encoding — registered variants.

``-smoke`` scores a 2000-vertex random subset. It exists so the pipeline can be
exercised in a couple of minutes; its number is not comparable to the full run.
"""

from brainscore import benchmark_registry

from .benchmark import LeBel2023Encoding

benchmark_registry['LeBel2023-UTS03-encoding'] = lambda: LeBel2023Encoding()
benchmark_registry['LeBel2023-UTS03-encoding-smoke'] = lambda: LeBel2023Encoding(
    identifier='LeBel2023-UTS03-encoding-smoke', max_targets=2000)
