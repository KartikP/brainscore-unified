"""The paper's memo RSA model, loaded only inside the Python 3.12 worker."""

from enum import IntEnum

import jax
from memo import memo


class U(IntEnum):
    GREEN = 0b0001
    PINK = 0b0010
    SQUARE = 0b0100
    ROUND = 0b1000


class R(IntEnum):
    GREEN_SQUARE = U.GREEN | U.SQUARE
    GREEN_CIRCLE = U.GREEN | U.ROUND
    PINK_CIRCLE = U.PINK | U.ROUND


def denotes(utterance, referent):
    return (utterance & referent) != 0


@memo
def L[u: U, r: R](beta, depth):
    listener: thinks[
        speaker: chooses(r_true in R, wpp=1),
        speaker: chooses(u_said in U, wpp=
            denotes(u_said, r_true) *
            (1 if depth == 0 else exp(beta * L[u_said, r_true](beta, depth - 1)))
        ),
    ]
    listener: observes [speaker.u_said] is u
    listener: chooses(r_pick in R, wpp=Pr[speaker.r_true == r_pick])
    return Pr[listener.r_pick == r]


def predict(*, depth: int, beta: float,
            utterances: list[str]) -> tuple[list[list[float]], str]:
    if type(depth) is not int or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError('beta must be numeric')
    if not utterances or any(not isinstance(item, str)
                             or item.upper() not in U.__members__
                             for item in utterances):
        raise ValueError('utterances must be known nonempty utterance labels')
    table = jax.device_get(L(float(beta), depth))
    rows = [table[U[name.upper()].value.bit_length() - 1].tolist()
            for name in utterances]
    return rows, str(table.dtype)
