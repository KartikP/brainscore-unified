"""BrainScoreModel is a Subject; core has no other base class."""


from brainscore import BrainScoreModel
from brainscore_core.contract import Subject


class TestSubjectRename:


    def test_brainscoremodel_subclasses_subject(self):
        assert issubclass(BrainScoreModel, Subject)
        assert BrainScoreModel.__bases__ == (Subject,)
