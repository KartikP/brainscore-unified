"""Public CogGym experiments, evaluated by the pinned upstream runner."""
from functools import partial

from brainscore import benchmark_registry

# Matches CogGym a1cd9df1118fec80eba7463de237d1497d77e041's public manifest.
EXPERIMENTS = {
    **{f'Levine2020Logic/exp{i}': 'text' for i in range(1, 7)},
    'Yoon2020Polite/exp1': 'text',
    'Yoon2020Polite/exp2': 'text',
    'Tsvilodub2025Nonliteral/exp1': 'text',
    'Tsvilodub2025Nonliteral/exp3b': 'text',
    'Hu2023Fine/exp1': 'text',
    'Aboody2025Inferring/exp1': 'image',
    'Aboody2025Inferring/exp2': 'image',
    'JaraEttinger2021Quantitative/exp1': 'image',
    'JaraEttinger2021Quantitative/exp2': 'image',
    'JaraEttinger2021Quantitative/exp3': 'image',
    'Chandra2024Cooperative/exp1': 'image',
    'Bass2022Partial/exp1': 'video',
    'Fu2025Hierarchical/exp1': 'video',
    'Fu2025Hierarchical/exp2': 'video',
    'sosa2021Moral/exp1': 'video',
    'sosa2021Moral/exp2': 'video',
    'sosa2021Moral/exp3': 'video',
}


def _load(experiment: str, **configuration):
    from .benchmark import CogGymBenchmark
    return CogGymBenchmark(experiment, **configuration)


for _experiment in EXPERIMENTS:
    benchmark_registry[f'CogGym.{_experiment.replace("/", ".")}'] = partial(_load, _experiment)
