"""A plain Python RSA reference for the notebook's implementation comparison."""

import math


UTTERANCES = ('green', 'pink', 'square', 'round')
CHOICES = ('green square', 'green circle', 'pink circle')
MEANINGS = {
    'green': {'green square', 'green circle'},
    'pink': {'pink circle'},
    'square': {'green square'},
    'round': {'green circle', 'pink circle'},
}


def listener_table(*, depth: int, beta: float = 1.0) -> dict[str, dict[str, float]]:
    """Infer referents with the paper's probability utility and uniform prior.

    At depth zero the speaker chooses uniformly among true utterances.
    At later depths its utility is the previous listener's success probability.
    This deliberately uses Python floats, independently of memo and JAX.
    """
    if type(depth) is not int or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError('beta must be numeric')
    if not math.isfinite(beta):
        raise ValueError('beta must be finite')
    previous = listener_table(depth=depth - 1, beta=beta) if depth else None
    speaker = {}
    for referent in CHOICES:
        weights = {
            utterance: (0.0 if referent not in MEANINGS[utterance] else
                        1.0 if previous is None else
                        math.exp(beta * previous[utterance][referent]))
            for utterance in UTTERANCES
        }
        total = math.fsum(weights.values())
        speaker[referent] = {word: weight / total for word, weight in weights.items()}

    # The uniform referent prior cancels in Bayes' rule.
    return {
        utterance: {
            referent: speaker[referent][utterance] / math.fsum(
                speaker[other][utterance] for other in CHOICES)
            for referent in CHOICES
        }
        for utterance in UTTERANCES
    }
