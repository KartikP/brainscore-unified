"""The minimal subject and the legacy compatibility base remain interoperable."""


from brainscore import BrainScoreModel
from brainscore_core.model_interface import Subject, UnifiedModel


class TestSubjectRename:


    def test_brainscoremodel_subclasses_subject(self):
        assert issubclass(BrainScoreModel, Subject)
        assert issubclass(BrainScoreModel, UnifiedModel)  # via the compatibility base
