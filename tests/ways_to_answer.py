"""The table of registered models by ways to answer, built with stand-in parts.

Run it in its own process (it replaces ``transformers``, the extractors and the
checkpoint loaders for the whole process):

    python -m tests.ways_to_answer docs/reference/ways_to_answer.md     # the reference page
    python -m tests.ways_to_answer --json                               # the rows, for a test

Every identifier in ``model_registry`` is built through its registry loader. No weights,
no processor files and no network: ``transformers`` gives stand-in networks, the extractors
are stand-ins that compute nothing, the download checks pass, and a socket connection
raises. The table holds only declared facts: inputs, regions, the ways the model has
(writing, likelihood, readout) and, for each kind of answer, the way that gives it.
"""
import importlib.abc
import importlib.machinery
import json
import socket
import sys
import types
from unittest import mock

REGENERATE = 'python -m tests.ways_to_answer docs/reference/ways_to_answer.md'
WAYS = {'generation': 'writing', 'likelihood': 'likelihood', 'readout': 'readout'}
# The towers whose layer names decide which extractor records a region: a registration whose
# regions span two input families without naming the family needs the real layer names.
# Depth 64 covers every registered layer; depth is not a declared fact.
SKELETONS = {
    'Qwen2_5_VLForConditionalGeneration': ('model.visual.blocks', 'model.language_model.layers'),
    'Qwen3_5ForConditionalGeneration': ('model.visual.blocks', 'model.language_model.layers'),
    'Blip2ForConditionalGeneration': ('vision_model.encoder.layers', 'language_model.model.decoder.layers'),
    'Wav2Vec2Model': ('encoder.layers',),
}
VJEPA1_BACKBONE = ('blocks',)


def _no_network(*args, **kwargs):
    raise RuntimeError('the table is built without a network')


def _network_class():
    import torch

    class Network(torch.nn.Module):
        """A stand-in network: one parameter, and a child module for every attribute asked."""

        def __init__(self, *args, skeleton=None, **kwargs):
            super().__init__()
            # Set in the dictionaries: the setters ask hasattr, which would make a child here.
            self._parameters['weight'] = torch.nn.Parameter(torch.zeros(1))
            self._listed = False                 # True: its layer names are real, an extractor lists them
            for path in SKELETONS.get(type(self).__name__, ()) if skeleton is None else skeleton:
                parent = self
                parent._listed = True
                for name in path.split('.'):
                    parent = getattr(parent, name)
                    parent._listed = True
                for index in range(64):
                    parent._modules[str(index)] = torch.nn.Identity()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name.startswith('_') or name == 'hf_device_map':   # a network loaded without a device map has none
                    raise
                child = Network()
                self._modules[name] = child
                return child
    return Network


class _Transformers(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``transformers`` and its submodules: every name is a stand-in network class."""

    def __init__(self, network):
        self.network = network

    def find_spec(self, name, path, target=None):
        if name == 'transformers' or name.startswith('transformers.'):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        def attribute(name):
            if name.startswith('__'):
                raise AttributeError(name)
            return type(name, (self.network,), {})
        module.__getattr__ = attribute
        return module

    def exec_module(self, module):
        pass


def _extractor_class(network):
    class Extractor:
        """A stand-in extractor: it lists the layers of the network it wraps when they are real."""

        def __init__(self, *args, **kwargs):
            self.identifier = kwargs.get('identifier', 'stand-in')
            model = kwargs.get('model', args[0] if args else None)
            if model is not None and not (isinstance(model, network) and not model._listed):
                names = [name for name, _ in model.named_modules() if name]
                self.layer_names = lambda: names

        def __call__(self, *args, **kwargs):
            raise AssertionError('a stand-in extractor computes nothing')
    return Extractor


def install_stand_ins():
    """Replace every heavy part in this process. Call before ``import brainscore``."""
    socket.socket.connect = _no_network
    network = _network_class()
    sys.modules.pop('transformers', None)
    sys.meta_path.insert(0, _Transformers(network))
    extractor = _extractor_class(network)
    import brainscore_core.hf_compat as hf_compat
    hf_compat.pin_image_processor = lambda processor, *args, **kwargs: processor
    import brainscore.models._downloads as downloads
    downloads.hf_preflight = lambda *args, **kwargs: {}
    for module, name in [('brainscore.model_helpers.text_wrapper', 'TextWrapper'),
                         ('brainscore.model_helpers.vlm_vision_wrapper', 'VLMVisionWrapper'),
                         ('brainscore.model_helpers.vlm_video_wrapper', 'VLMVideoWrapper'),
                         ('brainscore.model_helpers.video_wrapper', 'VideoWrapper'),
                         ('brainscore.model_helpers.audio_wrapper', 'AudioWrapper'),
                         ('brainscore_vision.model_helpers.activations.pytorch', 'PytorchWrapper')]:
        setattr(importlib.import_module(module), name, extractor)
    importlib.import_module('brainscore_vision.model_helpers.activations.pca').LayerPCA = mock.MagicMock()
    # The checkpoint loaders that read a file of their own (not through transformers).
    vjepa1 = importlib.import_module('brainscore.models.vjepa_v1.model')
    vjepa1.load_backbone = lambda identifier: network(skeleton=VJEPA1_BACKBONE)
    qwen25 = importlib.import_module('brainscore.models.qwen25_vl_3b.model')
    qwen25._bf16_provenance = lambda *args: {'checkpoint': 'stand-in'}     # hashes checkpoint files
    # A registration made with the builder states its parts' paths: its stand-in network has
    # blocks there (no table by class name). Set before a plugin's model.py imports the builder.
    vlm = importlib.import_module('brainscore.model_helpers.vlm')
    build = vlm.build_vlm_subject

    def on_stated_paths(*args, **kwargs):
        parts = [kwargs[name] for name in ('vision', 'language', 'audio') if kwargs.get(name)]
        vlm._load = lambda *a: (network(skeleton=[f'{part[0]}.{part[1]}' for part in parts]), network())
        return build(*args, **kwargs)
    vlm.build_vlm_subject = on_stated_paths


def _row(subject):
    """The declared facts of one subject."""
    outputs = sorted(subject.out_channels)
    kinds = [channel.split(':', 1)[1] for channel in outputs if channel.startswith('behavior:')]
    from brainscore import BrainScoreModel
    subject = getattr(subject, '_subject', subject)    # a subject that adds provenance to another
    if not isinstance(subject, BrainScoreModel):       # a subject with its own code, no parts
        return {'inputs': sorted(subject.in_channels),
                'regions': [c.split(':', 1)[1] for c in outputs if c.startswith('neural:')],
                'ways': [], 'answers': {kind: 'own code' for kind in kinds}}
    from brainscore._model.behavioral import _formats_by_part
    parts = _formats_by_part(subject)
    if not subject.extractor:
        parts.pop('readout', None)         # a readout layer without an extractor never answers
    owners = subject._behavioral.owners    # the part that gives each kind (rule 4.11)
    answers = {}
    for kind in kinds:
        # rt comes with probabilities, from the same part.
        owner = owners.get(f'behavior:{kind}') or owners.get('behavior:probabilities')
        answers[kind] = WAYS[owner]
    return {'inputs': sorted(subject.in_channels),
            'regions': [c.split(':', 1)[1] for c in outputs if c.startswith('neural:')],
            'ways': [WAYS[part] for part in ('generation', 'likelihood', 'readout') if part in parts],
            'answers': answers}


def build_rows():
    """Each registered identifier with its declared facts, or the error that stopped its build."""
    install_stand_ins()
    from brainscore import model_registry
    rows = {}
    for identifier in sorted(model_registry):
        try:
            rows[identifier] = _row(model_registry[identifier]())
        except Exception as error:         # recorded, so one broken loader does not hide the others
            rows[identifier] = {'error': f'{type(error).__name__}: {str(error).splitlines()[0][:200]}'}
    return rows


def _cell(values):
    return ', '.join(values) if values else '-'


def markdown(rows):
    """The page: two sentences, the table (identifiers with equal facts share a row), the command."""
    groups = {}
    for identifier, row in rows.items():
        groups.setdefault(json.dumps(row, sort_keys=True), []).append(identifier)
    lines = [
        '# Ways to answer',
        '',
        'Each registered model answers with the parts its registration wires: writing (a reply it '
        'generates), likelihood (the log probability of each option) or a readout (a classifier fitted '
        'on one of its layers). The table lists what each registration declares, and for each kind '
        'of answer the one part that gives it.',
        '',
        '| Models | Inputs | Regions | Ways | Kinds of answers (the way that gives each) |',
        '| --- | --- | --- | --- | --- |',
    ]
    for key, identifiers in sorted(groups.items(), key=lambda item: sorted(item[1])[0]):
        row = json.loads(key)
        models = ', '.join(f'`{identifier}`' for identifier in sorted(identifiers))
        if 'error' in row:
            lines.append(f'| {models} | not built: {row["error"]} | | | |')
            continue
        answers = _cell([f'{kind} ({way})' for kind, way in row['answers'].items()])
        lines.append(f'| {models} | {_cell(row["inputs"])} | {_cell(row["regions"])} | '
                     f'{_cell(row["ways"])} | {answers} |')
    lines += ['',
              '"own code": the subject is not a `BrainScoreModel`; its own `participate` gives the '
              'answer (`chance-baseline` gives uniform probabilities).',
              '',
              "A model that answers through a provider's API builds without a key and without a "
              "network; it needs the provider's key only to answer.",
              '',
              'The table is generated from the registrations with stand-in parts (no weights, no '
              'network). Regenerate it from the folder `unified`:',
              '',
              '```bash',
              REGENERATE,
              '```',
              '']
    return '\n'.join(lines)


def main(argv):
    rows = build_rows()
    if argv[:1] == ['--json']:
        print(json.dumps(rows, indent=1, sort_keys=True))
        return
    page = markdown(rows)
    if argv:
        with open(argv[0], 'w') as file:
            file.write(page)
    else:
        print(page)


if __name__ == '__main__':
    main(sys.argv[1:])
