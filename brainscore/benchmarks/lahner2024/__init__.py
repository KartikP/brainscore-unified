from brainscore import benchmark_registry
from .benchmark import Lahner2024BOLDMoments

benchmark_registry['Lahner2024-fMRI-naturalistic'] = Lahner2024BOLDMoments
