import pytest

from brainscore import BrainScoreModel


def test_brainscore_model_gets_channel_identity_from_preprocessors():
    model = BrainScoreModel(
        identifier="brain-score-model",
        model=None,
        region_layer_map={"IT": "layer4"},
        preprocessors={"vision": lambda *args, **kwargs: None},
        behavioral_readout_layer="layer4",
    )

    assert model.in_channels == {"vision:image", "control"}   # a vision key takes stills
    assert model.out_channels == {"neural:IT", "behavior", "behavior:choice",
                                  "behavior:probabilities", "behavior:rating", "behavior:rt"}
    assert model.required_channels == set()


def test_region_map_without_extractor_declares_no_neural_output():
    """Rule 2.4: a declared channel must work; a map alone cannot record."""
    from brainscore import BrainScoreModel
    model = BrainScoreModel('unwired', region_layer_map={'IT': 'layer'})
    assert 'neural:IT' not in model.out_channels
    with pytest.raises(NotImplementedError, match='No session handler'):
        model.check_session_support({'neural:IT'})
    wired = BrainScoreModel('wired', region_layer_map={'IT': 'layer'},
                            extractor={'vision': lambda stimuli, layers=None, **k: None})
    assert 'neural:IT' in wired.out_channels
