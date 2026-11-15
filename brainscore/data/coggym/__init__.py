"""CogGym: the authors' pinned checkout, a local asset (see README.md).

The runner reads experiments, human results and scoring code from that
checkout; this plugin registers no dataset.
"""
from brainscore.data import local

COGGYM_CHECKOUT = local.register(local.LocalAsset(
    name='coggym-checkout',
    env_var='BRAINSCORE_COGGYM_CHECKOUT',
    default_path='coggym',
    kind='stimuli',
    why_local="CogGym's runner reads its experiments, human results and scoring code "
              "from an unmodified Git checkout at a pinned revision.",
    source='https://github.com/lance-ying/coggym (revision a1cd9df1118fec80eba7463de237d1497d77e041)',
    obtain='git clone https://github.com/lance-ying/coggym.git, then check out the revision above.',
    used_by=[f'CogGym.{experiment}' for experiment in (
        'Levine2020Logic.exp1', 'Levine2020Logic.exp2', 'Levine2020Logic.exp3',
        'Levine2020Logic.exp4', 'Levine2020Logic.exp5', 'Levine2020Logic.exp6',
        'Yoon2020Polite.exp1', 'Yoon2020Polite.exp2',
        'Tsvilodub2025Nonliteral.exp1', 'Tsvilodub2025Nonliteral.exp3b', 'Hu2023Fine.exp1',
        'Aboody2025Inferring.exp1', 'Aboody2025Inferring.exp2',
        'JaraEttinger2021Quantitative.exp1', 'JaraEttinger2021Quantitative.exp2',
        'JaraEttinger2021Quantitative.exp3', 'Chandra2024Cooperative.exp1',
        'Bass2022Partial.exp1', 'Fu2025Hierarchical.exp1', 'Fu2025Hierarchical.exp2',
        'sosa2021Moral.exp1', 'sosa2021Moral.exp2', 'sosa2021Moral.exp3')],
))
