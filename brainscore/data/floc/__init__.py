"""Register the fLoc localizer stimuli (Stigliani et al. 2015); no I/O at import."""
from brainscore import data_builder_registry, stimulus_set_registry
from brainscore.data.packaging import DataBuilder

IDENTIFIER = 'fLoc'

data_builder_registry['floc'] = lambda: DataBuilder(
    source='fLoc (Stigliani et al. 2015), https://github.com/VPNL/fLoc at a pinned commit. The repository '
           'holds no licence file; its README asks for an acknowledgement and the citation.',
    access='public, no account.',
    layout='none.',
    needs='git; about 190 MB.',
    writes='Nothing to package: the first `load_stimulus_set("fLoc")` clones the repository into '
           '`~/brainscore-data/sources/floc` and reads its images in place.',
)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


stimulus_set_registry[IDENTIFIER] = _stimuli
