"""The answer owners the removed fixed order (generation, likelihood, readout) chose.

A test oracle, written independently of the library: the outputs each answer
argument declares (rule 2.4) and the old order. A test that builds a model
with several answer parts states these owners, so it runs as before rule 4.11.
"""

# The answer outputs each constructor argument gives, and its part.
OUTPUTS = {
    'answer': ('generation', {'behavior', 'behavior:choice'}),
    'reply': ('generation', {'behavior:choice', 'behavior:rating', 'behavior:confidence',
                             'behavior:text'}),
    'likelihood': ('likelihood', {'behavior:choice', 'behavior:probabilities'}),
    'behavioral_readout_layer': ('readout', {'behavior', 'behavior:choice',
                                             'behavior:probabilities', 'behavior:rating'}),
}
ORDER = ('generation', 'likelihood', 'readout')


def fixed_order_owners(*arguments) -> dict:
    """For each output several of ``arguments`` give, the part the fixed order chose."""
    arguments = [name for name in arguments if name in OUTPUTS]
    owners = {}
    for output in set().union(*(OUTPUTS[name][1] for name in arguments)):
        parts = {OUTPUTS[name][0] for name in arguments if output in OUTPUTS[name][1]}
        if len(parts) > 1:
            owners[output] = next(part for part in ORDER if part in parts)
    return owners


def owned(*arguments) -> dict:
    """The ``capability_config`` keyword that states ``fixed_order_owners``."""
    return {'capability_config': {'behavior-sessions': fixed_order_owners(*arguments)}}
