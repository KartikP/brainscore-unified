"""CogGym: no dataset here (see README.md).

The CogGym benchmarks read experiments, human results and scoring code from the authors'
repository at a pinned commit, in a folder Brain-Score makes at the first use
(``brainscore/harnesses/_environment_folder.py``); this plugin registers no dataset, and
``package_data('coggym')`` says so.
"""
from brainscore import data_builder_registry
from brainscore.data.packaging import DataBuilder

data_builder_registry['coggym'] = lambda: DataBuilder(
    source='CogGym (Ying et al.), https://github.com/lance-ying/coggym at a pinned commit. The repository '
           'holds `LICENSE` (MIT) and `DATA_LICENSE.md`.',
    access='public, no account.',
    layout='none.',
    needs='git and the packages of `brainscore/harnesses/coggym/requirements.txt`, in a folder of its own.',
    writes='Nothing to package: the first `load_benchmark("CogGym...")` makes the checkout and its environment '
           '(see this plugin\'s README).',
)
