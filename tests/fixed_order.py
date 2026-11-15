"""The answer owners the removed fixed order (generation, likelihood, readout) chose.

A test oracle, written independently of the library: the outputs each answer
argument declares (rule 2.4) and the old order. A test that builds a model
with several answer parts states these owners, so it runs as before rule 4.11.
"""

# The answer outputs each constructor argument gives, and its part. Per output:
# None when the part gives it natively, else the conversion that derives it.
OUTPUTS = {
    'answer': ('generation', {'behavior': None, 'behavior:choice': None}),
    'reply': ('generation', {'behavior:choice': None, 'behavior:rating': None,
                             'behavior:confidence': None, 'behavior:text': None}),
    'likelihood': ('likelihood', {'behavior:choice': 'softmax_argmax',
                                  'behavior:probabilities': 'softmax'}),
    'behavioral_readout_layer': ('readout', {'behavior': None, 'behavior:choice': 'argmax',
                                             'behavior:probabilities': None,
                                             'behavior:rating': None}),
}
ORDER = ('generation', 'likelihood', 'readout')


def fixed_order_owners(*arguments) -> dict:
    """For each output several of ``arguments`` give, the statement of the part the fixed order chose.

    A part that gives the output natively is stated by its name; a part that
    derives it, as ``(part, conversion)``.
    """
    arguments = [name for name in arguments if name in OUTPUTS]
    owners = {}
    for output in set().union(*(OUTPUTS[name][1] for name in arguments)):
        parts = {OUTPUTS[name][0]: OUTPUTS[name][1][output] for name in arguments
                 if output in OUTPUTS[name][1]}
        if len(parts) > 1:
            part = next(part for part in ORDER if part in parts)
            owners[output] = part if parts[part] is None else (part, parts[part])
    return owners


def owned(*arguments) -> dict:
    """The ``capability_config`` keyword that states ``fixed_order_owners``."""
    return {'capability_config': {'behavior-sessions': fixed_order_owners(*arguments)}}
