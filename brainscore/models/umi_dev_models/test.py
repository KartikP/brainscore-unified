"""The study's models: offline on a stand-in of each model's stated layout (no weights: every real
``from_pretrained`` raises), and with the real weights in the tests marked ``slow`` (a GPU machine).

    pytest brainscore/models/umi_dev_models/test.py -m "not slow and not qualification"     # offline
    HF_HUB_OFFLINE=1 pytest brainscore/models/umi_dev_models/test.py -m qualification        # settings files, no weights
    BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1 pytest brainscore/models/umi_dev_models/test.py -m slow -k qwen3-vl-2b
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from brainscore import model_registry
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from . import model as study

IDENTIFIERS = sorted(study.MODEL_CONFIGS)
WORDS = ['[PAD]', '[UNK]', 'system', 'user', 'assistant', ':', '<image>', 'cat', 'dog', 'the', 'sat', 'Action', '1', '9']
REPLY = ['dog']    # the words the stand-in writes; a test may set another reply


# --- the stand-in ---------------------------------------------------------------------------

class _Part(torch.nn.Module):
    """A model part with four blocks at the stated block path; it runs them in order."""

    def __init__(self, blocks, words=False, tokens=None):
        super().__init__()
        self._tokens = tokens       # a picture as this many patch tokens (a square grid), for a pooled layer
        parent = self
        *names, last = blocks.split('.')
        for name in names:
            parent.add_module(name, torch.nn.Module())
            parent = getattr(parent, name)
        parent.add_module(last, torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(4)))
        self._path = blocks
        self.embed = torch.nn.Embedding(len(WORDS), 4) if words else None
        self.passes = 0

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, grid_thw=None, **_):
        self.passes += 1
        x = self.embed(input_ids) if self.embed is not None else (
            pixel_values.reshape(len(pixel_values), -1)[:, :4].to(next(self.parameters()).dtype))
        if self._tokens:
            x = x[:, None, :].expand(-1, self._tokens, -1)
        for block in self.get_submodule(self._path):
            x = torch.tanh(block(x))
        return SimpleNamespace(last_hidden_state=x)


class _StatedLayout(torch.nn.Module):
    """The parts at the paths a configuration states; pictures and clips go through the vision part."""

    def __init__(self, config):
        super().__init__()
        torch.manual_seed(0)
        for (root, blocks), words in ((config['vision'], False), (config['language'], True)):
            *names, last = root.split('.')
            parent = self
            for name in names:
                if name not in parent._modules:
                    parent.add_module(name, torch.nn.Module())
                parent = parent._modules[name]
            parent.add_module(last, _Part(blocks, words, None if words or not config.get('token_pool') else 16))
        self._vision, self._language = config['vision'][0], config['language'][0]
        self.head = torch.nn.Linear(4, len(WORDS))

    @property
    def device(self):
        return self.head.weight.device

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
                pixel_values_videos=None, video_grid_thw=None, **_):
        self.passes = getattr(self, 'passes', 0) + 1
        for given, grid in ((pixel_values, image_grid_thw), (pixel_values_videos, video_grid_thw)):
            if given is not None:
                self.get_submodule(self._vision)(pixel_values=given, grid_thw=grid)
        hidden = self.get_submodule(self._language)(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.head(hidden))

    def generate(self, input_ids, **inputs):
        self(input_ids=input_ids, **inputs)       # the prompt pass a recording keeps
        return torch.cat([input_ids, torch.tensor([[WORDS.index(word) for word in REPLY]])], dim=1)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                                eos_token='[PAD]', model_max_length=64)


def _image_pixels(images, return_tensors='pt', **settings):
    """Each picture as (3, 2, 2): its mean colour in every pixel (one whole image)."""
    return {'pixel_values': torch.stack([torch.tensor(np.asarray(image, np.float32).mean((0, 1)) / 255)
                                         [:, None, None].expand(3, 2, 2) for image in images])}


def _clip_patches(videos, video_metadata=None, return_tensors='pt'):
    """A clip as flattened patches: two frames per time step, one patch per step."""
    frames = torch.tensor(np.asarray(videos[0], np.float32)) / 255
    steps = len(frames) // 2
    patches = torch.stack([torch.cat([frames[2 * k:2 * k + 2].mean((0, 1, 2)), torch.ones(1)]) for k in range(steps)])
    return {'pixel_values_videos': patches, 'video_grid_thw': torch.tensor([[steps, 1, 1]])}


_clip_patches.temporal_patch_size, _clip_patches.fps = 2, 2


class FakeProcessor:
    """A chat processor with the keys of the Qwen3-VL processor: input_ids, attention_mask; with
    pictures pixel_values and image_grid_thw; with clips pixel_values_videos and video_grid_thw."""

    def __init__(self):
        self.tokenizer, self.image_processor, self.video_processor = _tokenizer(), _image_pixels, _clip_patches

    def apply_chat_template(self, messages, add_generation_prompt=False, **_):
        def words(content):
            return content if isinstance(content, str) else ' '.join(
                '<image>' if part['type'] in ('image', 'video') else part.get('text', '') for part in content)
        return ' '.join(f"{m['role']} : {words(m['content'])}" for m in messages) + (
            ' assistant :' if add_generation_prompt else '')

    def __call__(self, text, images=None, videos=None, return_tensors='pt', padding=True, **_):
        inputs = dict(self.tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False))
        if videos:
            inputs.update(_clip_patches(videos))
        if images:
            inputs['pixel_values'] = torch.tensor([[*(np.asarray(image, np.float32).mean((0, 1)) / 255), 1.0]
                                                   for image in images])
            inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]] * len(images))
        return transformers.BatchFeature(inputs)


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """Every model of the study on a stand-in of its stated layout, in its stated precision, with no
    checkpoint and no download; the pixel path's LayerPCA fits for real (two components, four pictures)."""
    from brainscore_vision.model_helpers.activations import pca

    def refuse(*args, **kwargs):
        raise AssertionError('a test must not load a checkpoint')
    monkeypatch.setattr(transformers.PreTrainedModel, 'from_pretrained', classmethod(refuse))
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    monkeypatch.setattr(study, '_refuse', lambda identifier, config: None)

    def load(identifier, config):
        return _StatedLayout(config).to(getattr(torch, config['dtype'])), FakeProcessor()
    monkeypatch.setattr(study, '_load', load)
    corpus = list(_pictures(tmp_path, n=4)['image_file_name'])
    monkeypatch.setattr(pca, '_get_imagenet_val', lambda num_images: corpus)
    hook = pca.LayerPCA.hook.__func__
    monkeypatch.setattr(pca.LayerPCA, 'hook', classmethod(lambda cls, extractor, n_components: hook(cls, extractor, 2)))
    return tmp_path


def _pictures(folder, n=2, labels=None):
    from PIL import Image
    paths = []
    for index in range(n):
        paths.append(str(folder / f'picture{index}.png'))
        Image.new('RGB', (8, 8), (40 * index, 90, 200)).save(paths[-1])
    stimuli = StimulusSet({'stimulus_id': [f'p{i}' for i in range(n)], 'image_file_name': paths,
                           **({'image_label': labels} if labels else {})})
    stimuli.identifier = f'umi-dev-pictures-{n}-{bool(labels)}'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], paths))    # the legacy image extractor reads these
    return stimuli


def _clip(folder):
    import cv2
    path = str(folder / 'clip.mp4')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (64, 64))   # real video processors need 32 or more
    for frame in range(4):
        writer.write(np.full((64, 64, 3), 60 * frame, np.uint8))
    writer.release()
    stimuli = StimulusSet({'stimulus_id': ['c0'], 'video_path': [path]})
    stimuli.identifier = 'umi-dev-clip'
    return stimuli


def _collect(subject, stimuli, outputs, task):
    session = stimulus_session(stimuli, list(outputs), task)
    subject.participate(session)
    return {channel: session.collect(channel) for channel in outputs}


# --- offline tests --------------------------------------------------------------------------

def test_the_depth_rule():
    vision = study.depth_layers([f'blocks.{i}' for i in range(32)], study.DEPTH_TABLE['vision'])
    assert vision == {'V1': 'blocks.3', 'V2': 'blocks.8', 'V4': 'blocks.16', 'IT': 'blocks.28', 'VWFA': 'blocks.28'}
    assert study.depth_layers([f'l{i}' for i in range(36)], study.DEPTH_TABLE['language']) == {'language_system': 'l23'}
    assert study.depth_layers([f'l{i}' for i in range(26)], {'V1': 0.10}) == {'V1': 'l3'}     # 2.5 rounds up


def test_the_token_pool_averages_blocks_of_the_patch_grid():
    grid = np.arange(16, dtype=np.float32).reshape(1, 16, 1)          # a 4 x 4 grid, row by row
    assert study._pool_tokens(grid, 2)[0, :, 0].tolist() == [2.5, 4.5, 10.5, 12.5]
    with pytest.raises(ValueError):
        study._pool_tokens(np.zeros((1, 12, 1), np.float32), 2)


def test_the_answer_path_gets_the_whole_image_settings_a_processor_drops():
    """A processor that drops a setting it does not declare (SmolVLM's do_image_splitting) and reads its
    image processor instead still gives one whole image when the model answers."""
    class Dropping:
        image_processor = SimpleNamespace(do_image_splitting=True)

        def __call__(self, **_):
            return 5 if self.image_processor.do_image_splitting else 1     # tiles for one picture
    assert study._StatedProcessor(Dropping(), {'do_image_splitting': False})(text=['x']) == 1


def test_every_identifier_is_registered_with_its_suffix():
    assert set(IDENTIFIERS) <= set(model_registry) and all(name.endswith('_UMI-dev') for name in IDENTIFIERS)


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_each_declared_output_comes_out_of_one_session(stand_in, identifier):
    """The declarations follow from the stated parts, and every declared output (and clips, when
    the model takes them) comes out of a small session through participate."""
    global REPLY
    from brainscore_core.events import Perturbation, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change
    config, subject = study.MODEL_CONFIGS[identifier], model_registry[identifier]()
    folder = stand_in
    pictures, fitting = _pictures(folder, 4), _pictures(folder, 4, labels=['cat', 'dog', 'cat', 'dog'])
    sentences = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat sat', 'the dog sat']})
    sentences.identifier = 'umi-dev-sentences'
    passive = TaskContext(task_type='passive')
    provenance = subject._provenance
    assert provenance['region_layer_map_status'].startswith(study.DEPTH_HYPOTHESIS)
    assert provenance['readout_layer'].startswith(subject.region_layer_map['IT'])
    assert provenance['revision'] == config['revision']
    assert {f'neural:{r}' for r in ('V1', 'V2', 'V4', 'IT', 'VWFA', 'language_system')} | {
        'behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities', 'behavior:rating',
        'behavior:rt', 'behavior:text', 'motor'} <= subject.out_channels
    assert subject.output_inputs['motor'] == {'observation'}
    assert ('vision:video' in subject.in_channels) == config['video']
    pixel = config['image_processor_kwargs'] is not None
    assert ('perturbation' in subject.out_channels) == (not pixel) == ('lesion' in subject.in_channels)
    assert subject.output_inputs['neural:language_system'] == {'text'}      # one session for regions, one for an answer

    for region in ('V1', 'V2', 'V4', 'IT', 'VWFA'):
        values = _collect(subject, pictures, [f'neural:{region}'], passive)[f'neural:{region}'].values
        assert values.shape[0] == 4 and values.dtype == np.float32 and np.isfinite(values).all(), region
        assert values.shape[1] == (2 if pixel else 4), region         # the fitted PCA's components, or the hidden width
    language = _collect(subject, sentences, ['neural:language_system'], passive)['neural:language_system']
    assert language.shape[0] == 2 and np.isfinite(language.values).all()
    if config['video']:
        assert _collect(subject, _clip(folder), ['neural:IT'], passive)['neural:IT'].shape[0] == 1

    choice = Choice(['cat', 'dog'])
    for channel, stimuli, task in [
            ('behavior:choice', pictures, TaskContext(response=choice)),
            ('behavior:text', sentences, TaskContext(response=Text(max_tokens=2))),
            ('behavior:probabilities', pictures, TaskContext(response=choice, fitting_stimuli=fitting)),
            ('behavior:rt', pictures, TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                                                  fitting_stimuli=fitting))]:
        answer = _collect(subject, stimuli, [channel], task)[channel]
        assert answer.shape[0] == len(stimuli), channel
    continuations = StimulusSet({'stimulus_id': ['t0'], 'sentence': ['the cat'], 'options': [('sat', 'dog')]})
    continuations.identifier = 'umi-dev-continuations'
    logp = _collect(subject, continuations, ['behavior:log_probabilities'],
                    TaskContext(response=Choice(labels='options', verbatim=True)))['behavior:log_probabilities']
    assert np.isfinite(logp.values).all()
    given = REPLY
    try:
        REPLY = ['1']
        rating = _collect(subject, sentences, ['behavior:rating'], TaskContext(response=Scale(1, 7, 1)))['behavior:rating']
        assert list(rating.values) == [1.0, 1.0]
        REPLY = ['9']
        outside = _collect(subject, sentences, ['behavior:rating'], TaskContext(response=Scale(1, 7, 1)))['behavior:rating']
        assert set(outside['response_state'].values) == {'invalid'}
        REPLY = ['Action', ':', '1']
        from brainscore.harnesses.grid_game import GridGameEnv, play_game
        assert set(play_game(subject, GridGameEnv(size=5, seed=1, max_steps=3))['actions']) == {1}
    finally:
        REPLY = given
    subject.reset()

    if not pixel:   # a lesion changes exactly the selected units and is gone after its undo
        intact = _collect(subject, pictures, ['neural:IT'], passive)['neural:IT'].values
        record = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer=subject.region_layer_map['IT'], indices=[0, 1]),
            perturbation=Perturbation(kind='zero')))
        lesioned = _collect(subject, pictures, ['neural:IT'], passive)['neural:IT'].values
        assert (lesioned[:, :2] == 0).all() and np.array_equal(lesioned[:, 2:], intact[:, 2:])
        apply_state_change(subject, StateChange(kind='reset', handle_id=record.handle_id))
        assert np.array_equal(_collect(subject, pictures, ['neural:IT'], passive)['neural:IT'].values, intact)


@pytest.mark.parametrize('identifier', ['smolvlm2-256m_UMI-dev', 'gemma-3-4b_UMI-dev'])
def test_two_regions_of_the_picture_path_share_one_pass_over_the_table(stand_in, monkeypatch, identifier):
    """The picture path records the layers of several regions in one forward pass over the table, and
    each region's output equals its output when it is requested alone."""
    import uuid
    seen = []
    load = study._load

    def counting_load(name, config):
        model, processor = load(name, config)
        vision = model.get_submodule(config['vision'][0])
        vision.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(1), with_kwargs=True)
        return model, processor
    monkeypatch.setattr(study, '_load', counting_load)
    subject = model_registry[identifier]()
    passive = TaskContext(task_type='passive')
    pictures = _pictures(stand_in, 3)
    pictures.identifier = f'umi-dev-together-{uuid.uuid4().hex}'     # a name no cache has seen
    _collect(subject, _pictures(stand_in, 3), ['neural:V1', 'neural:IT'], passive)    # the PCA fits happen here
    del seen[:]
    together = _collect(subject, pictures, ['neural:V1', 'neural:IT'], passive)
    assert len(seen) == 1
    for channel, values in together.items():
        assert np.array_equal(values.values, _collect(subject, pictures, [channel], passive)[channel].values)


QWEN_VIDEO = [i for i in IDENTIFIERS if study.MODEL_CONFIGS[i].get('video_sampling')]
_KEYS = ('fps', 'min_frames', 'max_frames', 'do_sample_frames')


def _frames_kept(n_frames, clip_hz, settings):
    """The frame count of the Qwen video processors' sampling (transformers sample_frames)."""
    if not settings['do_sample_frames']:
        return n_frames
    wanted = int(n_frames / clip_hz * settings['fps'])
    return min(min(max(wanted, settings['min_frames']), settings['max_frames']), n_frames)


@pytest.mark.parametrize('identifier', QWEN_VIDEO)
def test_a_clip_is_sampled_at_the_stated_rate(stand_in, monkeypatch, identifier):
    """The stated frame sampling reaches the processor and the provenance, and clips are recorded."""
    monkeypatch.setattr(_clip_patches, 'fps', None)
    monkeypatch.setattr(_clip_patches, 'do_sample_frames', False, raising=False)
    subject = model_registry[identifier]()
    stated = study.MODEL_CONFIGS[identifier]['video_sampling']
    assert {k: getattr(_clip_patches, k) for k in _KEYS} == {k: stated[k] for k in _KEYS}
    assert 'video_sampling' in subject._provenance and 'qwen-vl-utils' in subject._provenance['video_sampling']
    clip = _collect(subject, _clip(stand_in), ['neural:IT'], TaskContext(task_type='passive'))['neural:IT']
    assert clip.shape[0] == 1 and np.isfinite(clip.values).all()


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_the_language_layer_during_a_task_is_refused_before_compute(stand_in, identifier):
    """One session for regions, one for an answer: a session that asks for the language layer under a task
    (with or without an answer among its outputs; on a picture, a clip or text) is refused before any
    forward pass of the model, its vision part or its language part. Pre-flight accepts such pairs (the text
    input answers the language layer), so the refusal comes when the session starts."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError, check_compatibility
    config, subject = study.MODEL_CONFIGS[identifier], model_registry[identifier]()
    pictures = _pictures(stand_in, 2)
    shown = {'image': StimulusSet({'stimulus_id': ['p0', 'p1'], 'sentence': ['the cat', 'the dog'],
                                   'image_file_name': list(pictures['image_file_name'])}),
             'text': StimulusSet({'stimulus_id': ['t0'], 'sentence': ['the cat']})}
    if config['video']:
        clip = _clip(stand_in)
        shown['video'] = StimulusSet({'stimulus_id': ['c0'], 'sentence': ['the cat'], 'video_path': list(clip['video_path'])})
    for name, stimuli in shown.items():
        stimuli.identifier = f'umi-dev-task-{name}'
    assert check_compatibility(subject, SimpleNamespace(
        identifier='picture-and-text', trial_memory='none', required_input_channels=frozenset({'vision:image', 'text'}),
        requested_output_channels=frozenset({'behavior:text', 'neural:language_system'}))).kind == 'full'
    network = subject._subject._model
    parts = [network, network.get_submodule(config['vision'][0]), network.get_submodule(config['language'][0])]
    for form, stimuli in shown.items():
        present = ('text',) if form == 'text' else (form, 'text')
        for outputs in (['behavior:text', 'neural:language_system'], ['neural:language_system']):
            with pytest.raises(ExecutionNotFeasibleError, match='language_system'):
                _collect(subject, stimuli, outputs, TaskContext(response=Text(max_tokens=2), present=present))
    assert [getattr(part, 'passes', 0) for part in parts] == [0, 0, 0]


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_the_readout_answers_from_pictures_only(stand_in, identifier):
    """Probabilities and response times come from a readout of the IT layer of the vision part, fitted on
    pictures: a task on text or on clips is N/A in pre-flight, and refused before any compute when a
    session asks for it directly."""
    from brainscore_core.compatibility import CompatibilityError, ExecutionNotFeasibleError, check_compatibility
    config, subject = study.MODEL_CONFIGS[identifier], model_registry[identifier]()
    fitting = _pictures(stand_in, 4, labels=['cat', 'dog', 'cat', 'dog'])
    sentences = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat', 'the dog']})
    sentences.identifier = 'umi-dev-readout-text'
    shown = {'text': sentences}
    if config['video']:
        shown['vision:video'] = _clip(stand_in)
    network = subject._subject._model
    parts = [network, network.get_submodule(config['vision'][0]), network.get_submodule(config['language'][0])]
    for output in ('behavior:probabilities', 'behavior:rt'):
        assert subject.output_inputs[output] == {'vision:image'}
        for channel, stimuli in shown.items():
            with pytest.raises(CompatibilityError):
                check_compatibility(subject, SimpleNamespace(
                    identifier='readout', trial_memory='none', required_input_channels=frozenset({channel}),
                    requested_output_channels=frozenset({output})))
            task = TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'), fitting_stimuli=fitting)
            with pytest.raises(ExecutionNotFeasibleError, match='pictures'):
                _collect(subject, stimuli, [output], task)
    assert [getattr(part, 'passes', 0) for part in parts] == [0, 0, 0]


@pytest.mark.parametrize('output', ['behavior:probabilities', 'behavior:rt'])
@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_the_readout_answers_a_picture_with_other_inputs_or_without_a_table(stand_in, identifier, output):
    """A picture that reaches the readout is answered: next to inputs the task does not present or the
    model does not take in, and in a session that holds its events and no table. Pre-flight agrees; a
    session of text events and no table is refused before compute."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError, check_compatibility
    from brainscore_core.streaming import Session
    subject = model_registry[identifier]()
    fitting = _pictures(stand_in, 4, labels=['cat', 'dog', 'cat', 'dog'])
    pictures = _pictures(stand_in, 2)
    with_text = StimulusSet({**{c: list(pictures[c]) for c in pictures.columns}, 'sentence': ['the cat', 'the dog']})
    with_sound = StimulusSet({**{c: list(pictures[c]) for c in pictures.columns},
                              'audio_path': [str(stand_in / 'none.wav')] * 2})
    for stimuli, name in ((with_text, 'text'), (with_sound, 'sound')):
        stimuli.identifier, stimuli.stimulus_paths = f'umi-dev-pictures-and-{name}', pictures.stimulus_paths
    task = lambda present=None: TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                                            fitting_stimuli=fitting, present=present)
    for stimuli, present, inputs in ((pictures, None, {'vision:image'}),
                                     (with_text, ('image',), {'vision:image', 'text'}),
                                     (with_sound, ('image',), {'vision:image', 'audio'})):
        assert check_compatibility(subject, SimpleNamespace(
            identifier='readout', trial_memory='none', required_input_channels=frozenset(inputs),
            requested_output_channels=frozenset({output}))).kind in ('full', 'partial')
        assert len(_collect(subject, stimuli, [output], task(present))[output]) == 2
    files = StimulusSet({'stimulus_id': list(pictures['stimulus_id']), 'image_path': list(pictures['image_file_name'])})
    files.identifier = 'umi-dev-picture-events'
    events = list(iter(stimulus_session(files, [output], task()).source.next_input, None))
    session = Session(events, [output], task())
    subject.participate(session)
    assert len(session.collect(output)) == 2
    sentences = StimulusSet({'stimulus_id': ['t0'], 'sentence': ['the cat']})
    sentences.identifier = 'umi-dev-text-events'
    text = list(iter(stimulus_session(sentences, [output], task()).source.next_input, None))
    with pytest.raises(ExecutionNotFeasibleError, match='no picture'):
        subject.participate(Session(text, [output], task()))


def test_a_pinned_revision_reaches_every_load(monkeypatch):
    """The revision a model states reaches the download check, its processor and its weights."""
    import brainscore.models._downloads as downloads
    import brainscore_core.hf_compat as hf_compat
    live, seen = sys.modules['transformers'], []

    def preflight(identifier, hf_id, approximate_gb, revision=None):
        seen.append(('download check', revision))
        return {'local_files_only': True, **({'revision': revision} if revision else {})}

    class Weights(torch.nn.Linear):
        @classmethod
        def from_pretrained(cls, hf_id, **settings):
            seen.append(('weights', settings.get('revision')))
            return torch.nn.Linear(1, 1)
    monkeypatch.setattr(downloads, 'hf_preflight', preflight)
    monkeypatch.setattr(live.AutoProcessor, 'from_pretrained',
                        lambda hf_id, **settings: seen.append(('processor', settings.get('revision'))) or FakeProcessor())
    monkeypatch.setattr(hf_compat, 'pin_image_processor', lambda processor, *a, **k: processor)
    # The first lazy loads above can put another module object under the name: set the class on the one there now.
    monkeypatch.setattr(sys.modules['transformers'], 'PinnedWeights', Weights, raising=False)
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    pinned = {i: c for i, c in study.MODEL_CONFIGS.items() if c['revision'] != 'not pinned'}
    assert pinned and all(len(c['revision']) == 40 for c in pinned.values())     # a commit of the Hub files
    for identifier, config in pinned.items():
        seen.clear()
        study._load(identifier, dict(config, model_class='PinnedWeights'))
        assert seen == [('download check', config['revision']), ('processor', config['revision']),
                        ('weights', config['revision'])], identifier


@pytest.mark.parametrize('processor, whole, tiled', [
    ('SmolVLMImageProcessor', {'do_image_splitting': False}, {'do_image_splitting': True}),
    ('Gemma3ImageProcessor', {'do_pan_and_scan': False}, {'do_pan_and_scan': True, 'pan_and_scan_min_crop_size': 8,
                                                           'pan_and_scan_max_num_crops': 4,
                                                           'pan_and_scan_min_ratio_to_activate': 1.0}),
    ('GotOcr2ImageProcessor', {'crop_to_patches': False}, {'crop_to_patches': True, 'min_patches': 2, 'max_patches': 6}),
])
def test_one_whole_image_per_picture_or_a_refusal(tmp_path, processor, whole, tiled):
    """The installed image processors of the families: the stated settings give one whole image."""
    from PIL import Image
    small = {'SmolVLMImageProcessor': dict(size={'longest_edge': 96}, max_image_size={'longest_edge': 32}),
             'Gemma3ImageProcessor': dict(size={'height': 32, 'width': 32}),
             'GotOcr2ImageProcessor': dict(size={'height': 32, 'width': 32})}
    image_processor = getattr(sys.modules['transformers'], processor)(**small[processor])
    paths = []
    for index in range(2):
        paths.append(str(tmp_path / f'wide{index}.png'))
        Image.new('RGB', (96, 32), (40 * index, 90, 200)).save(paths[-1])
    assert study._whole_images(paths, image_processor, whole).shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match='not one whole image'):
        study._whole_images(paths, image_processor, tiled)


def test_a_part_offloaded_or_spread_is_refused():
    part = torch.nn.Linear(2, 2).to('meta')
    with pytest.raises(ValueError, match='one accelerator'):
        study._one_device('m', part, 'vision')


def test_a_missing_package_and_a_missing_class_are_named_before_any_download(monkeypatch):
    """requirements.txt states each package once; a model that uses one names it, and a load of that model
    without it says what to install. A model that does not use it loads."""
    import importlib
    import importlib.util
    from packaging.requirements import Requirement
    from brainscore import LoadError
    assert all(Requirement(line) for line in study.REQUIREMENTS)        # each line is a valid requirement
    assert all(set(c['packages']) <= set(study.REQUIREMENTS) for c in study.MODEL_CONFIGS.values())
    real = importlib.util.find_spec
    missing = set(study.REQUIREMENTS)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a, **k: None if name in missing else real(name, *a, **k))
    for package in missing:     # the processor's own check, where transformers has one
        monkeypatch.setattr(importlib.import_module('transformers.utils'), f'is_{package}_available', lambda: False, raising=False)
    monkeypatch.setattr(study, '_load', lambda *a: pytest.fail('no load before the refusal'))
    for identifier, config in study.MODEL_CONFIGS.items():
        if config['packages']:
            with pytest.raises(LoadError, match=f"needs the package '{config['packages'][0]}', which is not installed. Next step"):
                model_registry[identifier]()
        else:
            study._refuse(identifier, config)       # a model that uses none of the packages loads without them
    with pytest.raises(LoadError, match='NoSuchVLM'):
        study._refuse('m', dict(study.MODEL_CONFIGS['qwen3-vl-2b_UMI-dev'], model_class='NoSuchVLM'))


# A small configuration of each real class, for its path check (no settings files, no weights).
SMALL_CONFIGS = {
    'Qwen3VLForConditionalGeneration': lambda t: t.Qwen3VLConfig(
        text_config=dict(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, rope_scaling={'rope_type': 'default', 'mrope_section': [8, 4, 4]}),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           deepstack_visual_indexes=[0, 1])),
    'Qwen3VLMoeForConditionalGeneration': lambda t: t.Qwen3VLMoeConfig(
        text_config=dict(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, rope_scaling={'rope_type': 'default', 'mrope_section': [8, 4, 4]}, num_experts=2,
                         num_experts_per_tok=1, moe_intermediate_size=32),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           deepstack_visual_indexes=[0, 1])),
    'Qwen2_5_VLForConditionalGeneration': lambda t: t.Qwen2_5_VLConfig(
        text_config=dict(vocab_size=300, num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2,
                         num_key_value_heads=1, rope_scaling={'type': 'mrope', 'mrope_section': [8, 8, 16]}),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           fullatt_block_indexes=[2])),
    'SmolVLMForConditionalGeneration': lambda t: t.SmolVLMConfig(
        text_config=dict(vocab_size=1000, pad_token_id=2, num_hidden_layers=3, hidden_size=64, intermediate_size=64,
                         num_attention_heads=2, num_key_value_heads=1),
        vision_config=dict(num_hidden_layers=2, hidden_size=32, intermediate_size=32, num_attention_heads=2)),
    'Gemma3ForConditionalGeneration': lambda t: t.Gemma3Config(
        text_config=dict(vocab_size=300, hidden_size=32, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16),
        vision_config=dict(hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                           image_size=28, patch_size=14), mm_tokens_per_image=4),
    'InternVLForConditionalGeneration': lambda t: t.InternVLConfig(
        text_config=dict(model_type='qwen2', vocab_size=300, hidden_size=32, intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=2, num_key_value_heads=1),
        vision_config=dict(hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                           image_size=[28, 28], patch_size=[14, 14])),
}
CLASSES = sorted({c['model_class'] for c in study.MODEL_CONFIGS.values()})


@pytest.mark.parametrize('model_class', CLASSES)
def test_the_stated_paths_are_block_lists_of_the_real_classes(model_class):
    """Each class of the values file, built from a small config on the meta device (no weights)."""
    if model_class not in SMALL_CONFIGS:
        pytest.fail(f'{model_class} has no small configuration for its path check: add one to SMALL_CONFIGS')
    live = sys.modules['transformers']
    network_class = getattr(live, model_class)
    with torch.device('meta'):
        network = network_class(SMALL_CONFIGS[model_class](live))
    for row in (c for c in study.MODEL_CONFIGS.values() if c['model_class'] == model_class):
        for root, blocks in (row['vision'], row['language']):
            assert isinstance(network.get_submodule(f'{root}.{blocks}'), torch.nn.ModuleList)


# --- with the real weights ------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_the_real_model(identifier, tmp_path):
    subject = model_registry[identifier]()
    neural = _collect(subject, _pictures(tmp_path, 2), ['neural:V1', 'neural:IT'], TaskContext(task_type='passive'))
    assert all(a.shape[0] == 2 and np.isfinite(a.values).all() for a in neural.values())
    sentences = StimulusSet({'stimulus_id': ['s0', 's1'], 'sentence': ['the cat sat on the mat', 'a dog runs fast']})
    sentences.identifier = f'{identifier}-sentences'
    assert _collect(subject, sentences, ['neural:language_system'], TaskContext(task_type='passive'))[
        'neural:language_system'].shape[0] == 2
    (tmp_path / 'fit').mkdir()
    probabilities = _collect(subject, _pictures(tmp_path, 2), ['behavior:probabilities'], TaskContext(
        response=Choice(['cat', 'dog']),
        fitting_stimuli=_pictures(tmp_path / 'fit', 4, labels=['cat', 'dog', 'cat', 'dog'])))['behavior:probabilities']
    assert probabilities.shape == (2, 2) and np.allclose(probabilities.values.sum(1), 1)
    assert _collect(subject, _pictures(tmp_path, 2), ['behavior:choice'],
                    TaskContext(response=Choice(['cat', 'dog'])))['behavior:choice'].shape[0] == 2


# --- qualification: the real processor and a tiny model of the real class (no weights) --------
#
# Needs the family's settings files in the Hugging Face cache (the command is in README.md); no
# weight file. Marked `qualification`: not part of the offline run of this file.

def _small(model_class, config):
    """The family's real configuration with few layers and a small width; the parts the processor
    and the forward depend on (patch and merge sizes, rope sections, token ids, vocabulary) stay."""
    text, vision = config.text_config, config.vision_config
    for key, value in dict(num_hidden_layers=2, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1).items():
        setattr(text, key, value)
    if getattr(text, 'layer_types', None):
        # One type for each layer kept: the cache has one layer for each type, and a cache with layers
        # that no pass fills is never ready, so Gemma 3 treats every generation step as the first.
        text.layer_types = text.layer_types[:text.num_hidden_layers]
    if model_class.startswith('Qwen3VL'):
        text.hidden_size, text.head_dim = 64, 128                # rope sections sum to head_dim / 2
        vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 4, 64, 2, 64
        vision.out_hidden_size, vision.deepstack_visual_indexes = 64, [1, 2]
        if 'Moe' in model_class:
            text.num_experts, text.num_experts_per_tok, text.moe_intermediate_size = 4, 2, 32
    elif model_class.startswith('Qwen2_5_VL'):
        text.hidden_size = 256                                   # 2 heads of 128: the rope sections
        vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 4, 64, 2, 64
        vision.out_hidden_size, vision.fullatt_block_indexes = 256, [3]
    elif model_class in ('SmolVLMForConditionalGeneration', 'Gemma3ForConditionalGeneration',
                         'InternVLForConditionalGeneration'):     # SigLIP-like vision parts
        text.hidden_size = 64
        if hasattr(text, 'head_dim'):
            text.head_dim = 32
        vision.num_hidden_layers, vision.hidden_size, vision.intermediate_size, vision.num_attention_heads = 4, 32, 32, 2
    else:
        pytest.fail(f'{model_class} has no reduction for its qualification: add one to _small')
    return config


# One model of each class of the values file: its smallest download.
QUALIFY = sorted(min((i for i, c in study.MODEL_CONFIGS.items() if c['model_class'] == model_class),
                     key=lambda i: study.MODEL_CONFIGS[i]['download_gb']) for model_class in CLASSES)


@pytest.mark.qualification
@pytest.mark.parametrize('identifier', QUALIFY)
def test_qualify_the_family(identifier, monkeypatch, tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.hf_compat import pin_image_processor
    from brainscore_vision.model_helpers.activations import pca
    live = sys.modules['transformers']
    config = study.MODEL_CONFIGS[identifier]

    def refuse(*args, **kwargs):
        raise AssertionError('a qualification must not load a checkpoint')
    monkeypatch.setattr(live.PreTrainedModel, 'from_pretrained', classmethod(refuse))
    # Selected explicitly, the qualification fails for anything it needs that is missing.
    study._refuse(identifier, config)       # a missing package: the clear message, as a failure
    at = {'local_files_only': True, **({} if config['revision'] == 'not pinned' else {'revision': config['revision']})}
    try:
        settings = live.AutoConfig.from_pretrained(config['hf_id'], **at)
        processor = pin_image_processor(live.AutoProcessor.from_pretrained(config['hf_id'], **at), config['hf_id'], **at)
    except OSError as error:
        pytest.fail(f"the settings files of {config['hf_id']} at {config['revision']} are not in the cache: fetch them "
                    f"with the command in README.md (a gated model needs its terms accepted and a token): {error}")
    network_class = getattr(live, config['model_class'])
    torch.manual_seed(0)
    network = network_class(_small(config['model_class'], settings)).to(getattr(torch, config['dtype'])).eval()
    monkeypatch.setattr(study, '_load', lambda *a: (network, processor))
    corpus = list(_pictures(tmp_path, 4)['image_file_name'])
    monkeypatch.setattr(pca, '_get_imagenet_val', lambda num_images: corpus)
    hook = pca.LayerPCA.hook.__func__
    monkeypatch.setattr(pca.LayerPCA, 'hook', classmethod(lambda cls, extractor, n_components: hook(cls, extractor, 2)))
    subject = model_registry[identifier]()

    # The regions are the stated depths of the real blocks.
    blocks = len(network.get_submodule('.'.join(config['vision'])))
    expected = study.depth_layers([f"{config['vision'][1]}.{i}" for i in range(blocks)], study.DEPTH_TABLE['vision'])
    assert {region: subject.region_layer_map[region] for region in expected} == expected
    # One whole image for each picture, by the real processor.
    from PIL import Image
    wide = str(tmp_path / 'wide.png')
    Image.new('RGB', (96, 48), (200, 40, 40)).save(wide)
    if config['image_processor_kwargs'] is not None:
        assert study._whole_images([wide, wide], processor.image_processor, config['image_processor_kwargs']).shape[0] == 2
    else:
        assert len(processor.image_processor(images=[Image.open(wide)], return_tensors='pt')['image_grid_thw']) == 1
    # The same picture when the model answers: one whole image, the pixels of the recording path.
    if config['image_processor_kwargs'] is not None:
        tall = str(tmp_path / 'tall.png')
        Image.new('RGB', (1024, 256), (30, 160, 90)).save(tall)
        reference = study._whole_images([tall], processor.image_processor, config['image_processor_kwargs'])
        vision_module = network.get_submodule(config['vision'][0])
        shown, forward = [], vision_module.forward

        def seen(pixel_values, *args, **kwargs):
            shown.append(pixel_values.detach().float().cpu().numpy())
            return forward(pixel_values, *args, **kwargs)
        monkeypatch.setattr(vision_module, 'forward', seen)
        one = StimulusSet({'stimulus_id': ['tall'], 'image_file_name': [tall]})
        one.identifier = 'umi-dev-qualify-tall'
        _collect(subject, one, ['behavior:text'], TaskContext(response=Text(max_tokens=1)))
        monkeypatch.setattr(vision_module, 'forward', forward)
        answered = shown[0].reshape(-1, *reference.shape[1:])
        assert answered.shape[0] == 1 and np.allclose(answered, reference.astype(answered.dtype), atol=0.05)
    # Each declared output through participate, exported as finite float32.
    pictures = _pictures(tmp_path, 4)
    fitting = _pictures(tmp_path / 'fit' if (tmp_path / 'fit').mkdir() is None else tmp_path, 4,
                        labels=['cat', 'dog', 'cat', 'dog'])
    sentences = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat sat', 'the dog sat']})
    sentences.identifier = 'umi-dev-qualify-sentences'
    passive = TaskContext(task_type='passive')
    for channel in sorted(c for c in subject.out_channels if c.startswith('neural:')):
        stimuli = sentences if channel == 'neural:language_system' else pictures
        values = _collect(subject, stimuli, [channel], passive)[channel].values
        assert values.dtype == np.float32 and np.isfinite(values).all() and values.shape[0] == len(stimuli), channel
    choice = Choice(['cat', 'dog'])
    for channel, stimuli, task in [
            ('behavior:choice', pictures, TaskContext(response=choice)),
            ('behavior:text', sentences, TaskContext(response=Text(max_tokens=4))),
            ('behavior:rating', sentences, TaskContext(response=Scale(1, 7, 1))),
            ('behavior:probabilities', pictures, TaskContext(response=choice, fitting_stimuli=fitting)),
            ('behavior:rt', pictures, TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                                                  fitting_stimuli=fitting))]:
        assert _collect(subject, stimuli, [channel], task)[channel].shape[0] == len(stimuli), channel
    continuations = StimulusSet({'stimulus_id': ['t0'], 'sentence': ['the cat'], 'options': [('sat', 'ran')]})
    continuations.identifier = 'umi-dev-qualify-continuations'
    assert np.isfinite(_collect(subject, continuations, ['behavior:log_probabilities'], TaskContext(
        response=Choice(labels='options', verbatim=True)))['behavior:log_probabilities'].values).all()
    from brainscore.harnesses.grid_game import GridGameEnv, play_game
    assert play_game(subject, GridGameEnv(size=5, seed=1, max_steps=2))['actions']
    if config['video']:
        assert _collect(subject, _clip(tmp_path), ['neural:IT'], passive)['neural:IT'].shape[0] == 1
    # A written answer with the language layer, on a picture and text: refused before compute.
    trials = StimulusSet({'stimulus_id': ['p0'], 'image_file_name': [pictures['image_file_name'][0]], 'sentence': ['the cat']})
    trials.identifier = 'umi-dev-qualify-mixed'
    mixed = TaskContext(response=Text(max_tokens=2), present=('image', 'text'))
    with pytest.raises(ExecutionNotFeasibleError):
        _collect(subject, trials, ['behavior:text', 'neural:language_system'], mixed)


@pytest.mark.parametrize('identifier', [next(i for i in QWEN_VIDEO if 'qwen2.5' in i),
                                        next(i for i in QWEN_VIDEO if 'qwen3' in i)])
def test_the_stated_sampling_keeps_the_frame_count(stand_in, monkeypatch, identifier):
    """A 16-frame clip at 30 Hz gives 4 frames; a long clip gives the count of the settings before the
    change (the processor's class defaults, with the study's rate and sampling on for Qwen2.5-VL)."""
    before = {'fps': 2.0, 'min_frames': 4, 'max_frames': 768, 'do_sample_frames': True}
    for key in _KEYS:
        monkeypatch.setattr(_clip_patches, key, None, raising=False)
    model_registry[identifier]()
    after = {k: getattr(_clip_patches, k) for k in _KEYS}
    assert _frames_kept(16, 30, after) == _frames_kept(16, 30, before) == 4
    assert _frames_kept(30 * 600, 30, after) == _frames_kept(30 * 600, 30, before)

