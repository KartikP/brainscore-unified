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


# One line for each experiment of EXPERIMENTS: the identifier and the experiment it runs.
benchmark_registry['CogGym.Levine2020Logic.exp1'] = partial(_load, 'Levine2020Logic/exp1')
benchmark_registry['CogGym.Levine2020Logic.exp2'] = partial(_load, 'Levine2020Logic/exp2')
benchmark_registry['CogGym.Levine2020Logic.exp3'] = partial(_load, 'Levine2020Logic/exp3')
benchmark_registry['CogGym.Levine2020Logic.exp4'] = partial(_load, 'Levine2020Logic/exp4')
benchmark_registry['CogGym.Levine2020Logic.exp5'] = partial(_load, 'Levine2020Logic/exp5')
benchmark_registry['CogGym.Levine2020Logic.exp6'] = partial(_load, 'Levine2020Logic/exp6')
benchmark_registry['CogGym.Yoon2020Polite.exp1'] = partial(_load, 'Yoon2020Polite/exp1')
benchmark_registry['CogGym.Yoon2020Polite.exp2'] = partial(_load, 'Yoon2020Polite/exp2')
benchmark_registry['CogGym.Tsvilodub2025Nonliteral.exp1'] = partial(_load, 'Tsvilodub2025Nonliteral/exp1')
benchmark_registry['CogGym.Tsvilodub2025Nonliteral.exp3b'] = partial(_load, 'Tsvilodub2025Nonliteral/exp3b')
benchmark_registry['CogGym.Hu2023Fine.exp1'] = partial(_load, 'Hu2023Fine/exp1')
benchmark_registry['CogGym.Aboody2025Inferring.exp1'] = partial(_load, 'Aboody2025Inferring/exp1')
benchmark_registry['CogGym.Aboody2025Inferring.exp2'] = partial(_load, 'Aboody2025Inferring/exp2')
benchmark_registry['CogGym.JaraEttinger2021Quantitative.exp1'] = partial(_load, 'JaraEttinger2021Quantitative/exp1')
benchmark_registry['CogGym.JaraEttinger2021Quantitative.exp2'] = partial(_load, 'JaraEttinger2021Quantitative/exp2')
benchmark_registry['CogGym.JaraEttinger2021Quantitative.exp3'] = partial(_load, 'JaraEttinger2021Quantitative/exp3')
benchmark_registry['CogGym.Chandra2024Cooperative.exp1'] = partial(_load, 'Chandra2024Cooperative/exp1')
benchmark_registry['CogGym.Bass2022Partial.exp1'] = partial(_load, 'Bass2022Partial/exp1')
benchmark_registry['CogGym.Fu2025Hierarchical.exp1'] = partial(_load, 'Fu2025Hierarchical/exp1')
benchmark_registry['CogGym.Fu2025Hierarchical.exp2'] = partial(_load, 'Fu2025Hierarchical/exp2')
benchmark_registry['CogGym.sosa2021Moral.exp1'] = partial(_load, 'sosa2021Moral/exp1')
benchmark_registry['CogGym.sosa2021Moral.exp2'] = partial(_load, 'sosa2021Moral/exp2')
benchmark_registry['CogGym.sosa2021Moral.exp3'] = partial(_load, 'sosa2021Moral/exp3')
