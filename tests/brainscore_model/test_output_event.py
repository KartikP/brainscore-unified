"""OutputEvent union contract.

`process()` returns one of a declared set of output shapes, symmetric with
InputEvent. StreamEvent carries extension-defined outputs without growing the
union or changing Subject for each new domain. These are introspection tests: no model
weights, no BrainIO import — so they run anywhere.
"""

from brainscore import BrainScoreModel
from brainscore_core.model_interface import OutputEvent


def test_brainscoremodel_process_returns_output_event():
    assert BrainScoreModel.process.__annotations__.get('return') is OutputEvent
