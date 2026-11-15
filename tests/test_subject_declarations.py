"""The five subjects that inherited the deleted base class keep every declaration.

Values pinned before the base class was deleted (core contract.UnifiedModel): each class now
inherits Subject and declares the same inputs, outputs, required inputs, output inputs and traits.
"""
import pytest

pytestmark = pytest.mark.unit

def _declarations(subject):
    return {'in_channels': sorted(subject.in_channels), 'out_channels': sorted(subject.out_channels),
            'required_channels': sorted(subject.required_channels),
            'output_inputs': {k: sorted(v) for k, v in sorted(dict(subject.output_inputs).items())},
            'traits': sorted(subject.traits)}


class _OldVisionModel:
    identifier = 'old-vision'
    class layer_model:
        region_layer_map = {'IT': 'layer4', 'V4': 'layer3'}
    def visual_degrees(self): return 8
    def look_at(self, stimuli, number_of_trials=1): raise AssertionError
    def start_recording(self, region, time_bins=None): pass
    def start_task(self, task, fitting_stimuli=None): pass


class _OldLanguageModel:
    region_layer_mapping = {'language_system': 'transformer.h.0'}
    def identifier(self): return 'old-language'


def subjects():
    from brainscore import BrainScoreModel
    from brainscore.harnesses.human import HumanSubject
    from brainscore.models.chance_baseline.model import ChanceBaseline
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter
    extractor = {'vision': lambda stimuli, layers=None, **k: None}
    return {
        'BrainScoreModel, extractor': BrainScoreModel('feat', None, {'IT': 'layer4'}, extractor=extractor),
        'BrainScoreModel, policy': BrainScoreModel('agent', policy=lambda step: None),
        'BrainScoreModel, readout': BrainScoreModel('clf', None, {'IT': 'layer4'}, extractor=extractor,
                                                    behavioral_readout_layer='layer4'),
        'HumanSubject, default': HumanSubject('human'),
        'HumanSubject, regions': HumanSubject('human', in_channels={'vision:image', 'text'},
                                              region_layer_map={'IT': 'probe'}),
        'ChanceBaseline': ChanceBaseline(),
        'VisionModelAdapter': VisionModelAdapter(_OldVisionModel()),
        'LanguageModelAdapter': LanguageModelAdapter(_OldLanguageModel()),
    }




EXPECTED = {
    "BrainScoreModel, extractor": {
        "in_channels": [
            "control",
            "vision:image"
        ],
        "out_channels": [
            "neural:IT"
        ],
        "output_inputs": {
            "neural:IT": [
                "vision:image"
            ]
        },
        "required_channels": [],
        "traits": []
    },
    "BrainScoreModel, policy": {
        "in_channels": [
            "control",
            "observation"
        ],
        "out_channels": [
            "motor"
        ],
        "output_inputs": {},
        "required_channels": [],
        "traits": []
    },
    "BrainScoreModel, readout": {
        "in_channels": [
            "control",
            "vision:image"
        ],
        "out_channels": [
            "behavior",
            "behavior:probabilities",
            "behavior:rating",
            "behavior:rt",
            "neural:IT"
        ],
        "output_inputs": {
            "neural:IT": [
                "vision:image"
            ]
        },
        "required_channels": [],
        "traits": []
    },
    "ChanceBaseline": {
        "in_channels": [
            "audio",
            "text",
            "vision:image",
            "vision:video"
        ],
        "out_channels": [
            "behavior",
            "behavior:choice"
        ],
        "output_inputs": {},
        "required_channels": [],
        "traits": []
    },
    "HumanSubject, default": {
        "in_channels": [
            "vision:image"
        ],
        "out_channels": [],
        "output_inputs": {},
        "required_channels": [],
        "traits": []
    },
    "HumanSubject, regions": {
        "in_channels": [
            "text",
            "vision:image"
        ],
        "out_channels": [
            "neural:IT"
        ],
        "output_inputs": {},
        "required_channels": [],
        "traits": []
    },
    "LanguageModelAdapter": {
        "in_channels": [
            "text"
        ],
        "out_channels": [
            "behavior",
            "neural:language_system"
        ],
        "output_inputs": {},
        "required_channels": [
            "text"
        ],
        "traits": []
    },
    "VisionModelAdapter": {
        "in_channels": [
            "vision:image"
        ],
        "out_channels": [
            "behavior",
            "behavior:choice",
            "behavior:probabilities",
            "neural:IT",
            "neural:V4"
        ],
        "output_inputs": {},
        "required_channels": [
            "vision:image"
        ],
        "traits": []
    }
}


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_the_declarations_are_those_before_the_base_class_was_deleted(name):
    assert _declarations(subjects()[name]) == EXPECTED[name]


def test_each_subject_inherits_subject_only():
    from brainscore_core.contract import Subject
    for name, subject in subjects().items():
        assert type(subject).__bases__ == (Subject,), name
