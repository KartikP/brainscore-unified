"""LeBel 2023 story-listening fMRI — pickle to Brain-Score assembly.

The source is a ``SimpleNeuroidAssembly`` pickle from the Huth-lab ``encoding``
package (as redistributed by LITCoder). It holds one subject listening to 25
spoken stories, with whole-cortex BOLD sampled every 2 s.

Two facts about the source drive this module:

1. The pickle references ``encoding.*`` classes that are not installed. It is
   read with an unpickler that substitutes permissive stand-ins, so no
   third-party package is required.
2. ``tr_times`` is 15 entries longer than ``brain_data`` for every story, split
   10 from the head and 5 from the tail, so ``brain_data[i]`` is the volume
   acquired at ``tr_times[i + 10]``. The count alone does not fix the direction —
   getting it backwards misaligns features against BOLD by five samples.

The conversion emits one stimulus row per TR. Each row carries the words spoken
in the seconds leading up to that TR, which is what a text model reads.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

TR_SEC = 2.0
# Ten TRs are dropped from the head of each story and five from the tail, so
# brain_data[i] is the volume acquired at tr_times[i + 10]. This is the split the
# reference pipeline hardcodes (`downsampled[10:-5]` in LITcoder's train_lebel),
# and it is not interchangeable with the other way round: assuming 5/10 puts the
# features five TRs early, which made the regression fit BOLD from words up to
# six seconds in the *future* and roughly halved every score.
TRIM_HEAD = 10    # TRs dropped from the start of each story
TRIM_TAIL = 5     # TRs dropped from the end of each story
DEFAULT_CONTEXT_TR = 5    # words from this many TRs before each sample

SUBJECT = 'UTS03'
IDENTIFIER = 'LeBel2023-UTS03'


def _pickle_path() -> Path:
    """Where the source pickle lives; override with BRAINSCORE_LEBEL_PICKLE."""
    return Path(os.environ.get(
        'BRAINSCORE_LEBEL_PICKLE',
        Path.home() / 'Downloads' / 'assembly_lebel_uts03.pkl')).expanduser()


def _cache_dir() -> Path:
    return Path(os.environ.get(
        'BRAINSCORE_LEBEL_CACHE',
        Path.home() / '.brainscore-umi-cache' / 'lebel2023')).expanduser()


class _Stub:
    """Stand-in for an ``encoding.*`` class, accepting whatever it is given.

    The pickle stores plain attribute dicts, so restoring them onto a bare
    object recovers every field without importing the original package.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {'_state': state})


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('encoding'):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def _read_pickle(path: Path):
    with open(path, 'rb') as handle:
        return _StubUnpickler(handle).load()


def _context_sentences(words, word_times, tr_times, context_sec):
    """One text string per TR: the words spoken in the window ending at that TR.

    Words are matched by onset time, so a sample's text is exactly what the
    listener had heard in the preceding ``context_sec`` seconds. Silent stretches
    yield an empty window, which is filled with a single period so downstream
    tokenizers always receive a non-empty string.
    """
    word_times = np.asarray(word_times, dtype=float)
    order = np.argsort(word_times)          # onset order is not guaranteed
    word_times = word_times[order]
    words = np.asarray(words, dtype=object)[order]

    # Window edges for every TR, then one binary search per edge.
    starts = np.searchsorted(word_times, tr_times - context_sec, side='left')
    stops = np.searchsorted(word_times, tr_times, side='right')

    sentences = []
    for start, stop in zip(starts, stops):
        chunk = [str(w).strip() for w in words[start:stop]]
        text = ' '.join(w for w in chunk if w)
        sentences.append(text if text else '.')
    return sentences


def build(pickle_path=None, context_window_tr: int = DEFAULT_CONTEXT_TR):
    """Convert the source pickle into a (StimulusSet, NeuroidAssembly) pair.

    Returns whole-cortex BOLD with no ROI mask applied: every measured vertex is
    a prediction target.
    """
    path = Path(pickle_path) if pickle_path else _pickle_path()
    if not path.exists():
        raise FileNotFoundError(
            f"LeBel pickle not found at {path}. Set BRAINSCORE_LEBEL_PICKLE to "
            f"its location.")
    source = _read_pickle(path)
    context_sec = context_window_tr * TR_SEC

    rows, blocks = [], []
    for story in source.stories:
        story_data = source.story_data[story]
        bold = np.asarray(story_data.brain_data)
        n_tr = bold.shape[0]

        # Recover the acquisition time of each retained volume.
        all_tr_times = np.asarray(story_data.tr_times, dtype=float)
        expected = n_tr + TRIM_HEAD + TRIM_TAIL
        if len(all_tr_times) != expected:
            raise ValueError(
                f"{story}: expected {expected} tr_times for {n_tr} volumes "
                f"(trim {TRIM_HEAD}/{TRIM_TAIL}), found {len(all_tr_times)}.")
        tr_times = all_tr_times[TRIM_HEAD:TRIM_HEAD + n_tr]

        sentences = _context_sentences(
            story_data.words, story_data.data_times, tr_times, context_sec)

        rows.append(pd.DataFrame({
            'story_id': story,
            'tr_index': np.arange(n_tr),
            'tr_time_sec': tr_times,
            'sentence': sentences,
        }))
        blocks.append(bold)

    stimulus_set = StimulusSet(pd.concat(rows, ignore_index=True))
    # Globally unique and stable across rebuilds — the model caches on this.
    stimulus_set['stimulus_id'] = [
        f"{row.story_id}_tr{row.tr_index:04d}"
        for row in stimulus_set.itertuples()]
    stimulus_set.identifier = f'{IDENTIFIER}-context{context_window_tr}tr'
    stimulus_set.stimulus_paths = {}     # text-only; no files to resolve

    data = np.concatenate(blocks, axis=0).astype(np.float32)
    n_neuroid = data.shape[1]
    assembly = NeuroidAssembly(
        data,
        dims=('presentation', 'neuroid'),
        coords={
            'stimulus_id': ('presentation', stimulus_set['stimulus_id'].values),
            'story_id': ('presentation', stimulus_set['story_id'].values),
            'tr_index': ('presentation', stimulus_set['tr_index'].values),
            'tr_time_sec': ('presentation', stimulus_set['tr_time_sec'].values),
            'neuroid_id': ('neuroid', [f'v{i:05d}' for i in range(n_neuroid)]),
            'vertex_index': ('neuroid', np.arange(n_neuroid)),
            'subject': ('neuroid', [SUBJECT] * n_neuroid),
        },
    )
    assembly.name = IDENTIFIER
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['tr_sec'] = TR_SEC
    return stimulus_set, assembly


def load(context_window_tr: int = DEFAULT_CONTEXT_TR, use_cache: bool = True):
    """``build`` with an on-disk cache, since unpickling costs minutes and ~3 GB.

    The cache key includes the context window, so changing it produces a
    separate file rather than silently reusing stale text.
    """
    cache = _cache_dir() / f'lebel_uts03_context{context_window_tr}tr.nc'
    if use_cache and cache.exists():
        import xarray as xr
        assembly = NeuroidAssembly(xr.open_dataarray(cache).load())
        stimulus_set = StimulusSet(pd.read_csv(cache.with_suffix('.csv')))
        stimulus_set.identifier = f'{IDENTIFIER}-context{context_window_tr}tr'
        stimulus_set.stimulus_paths = {}
        assembly.attrs['stimulus_set'] = stimulus_set
        assembly.attrs['tr_sec'] = TR_SEC
        return stimulus_set, assembly

    stimulus_set, assembly = build(context_window_tr=context_window_tr)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Two things block a direct write. netCDF attrs take only scalars and
        # arrays, so the attached stimulus set travels as the sidecar CSV. And
        # NeuroidAssembly gathers its coords into MultiIndexes on construction,
        # which netCDF cannot represent — flatten them back to plain coords.
        # Reading the file re-gathers them.
        writable = assembly.reset_index(['presentation', 'neuroid'])
        writable.attrs = {'tr_sec': TR_SEC}
        # Write then rename, so an interrupted write cannot leave a truncated
        # file that later calls would happily load as a valid cache.
        partial = cache.with_suffix('.nc.partial')
        writable.to_netcdf(partial)
        partial.replace(cache)
        stimulus_set.to_csv(cache.with_suffix('.csv'), index=False)
    return stimulus_set, assembly

DEFAULT_CONTEXT_WORDS = 32


def build_word_level(pickle_path=None, context_words: int = DEFAULT_CONTEXT_WORDS):
    """Convert the source pickle into word-level stimuli plus the TR grid.

    One row per spoken word, each carrying the running context that ends at that
    word, so a model's representation of the row is its representation *of that
    word in context*. The row also carries the word's onset time, which is what
    lets features be resampled onto the fMRI grid afterwards
    (:func:`brainscore_core.temporal.lanczos_downsample`).

    This is the alternative to collapsing each TR to a single context window up
    front: it keeps the several words that fall inside one TR distinct, which is
    the information last-token selection throws away.

    Returns ``(word_stimuli, assembly, tr_times_by_story)``.
    """
    path = Path(pickle_path) if pickle_path else _pickle_path()
    if not path.exists():
        raise FileNotFoundError(
            f"LeBel pickle not found at {path}. Set BRAINSCORE_LEBEL_PICKLE to "
            f"its location.")
    source = _read_pickle(path)

    rows, blocks, tr_times_by_story = [], [], {}
    for story in source.stories:
        story_data = source.story_data[story]
        bold = np.asarray(story_data.brain_data)
        n_tr = bold.shape[0]
        all_tr_times = np.asarray(story_data.tr_times, dtype=float)
        expected = n_tr + TRIM_HEAD + TRIM_TAIL
        if len(all_tr_times) != expected:
            raise ValueError(
                f"{story}: expected {expected} tr_times for {n_tr} volumes, "
                f"found {len(all_tr_times)}.")
        tr_times_by_story[story] = all_tr_times[TRIM_HEAD:TRIM_HEAD + n_tr]

        words = [str(w).strip() for w in story_data.words]
        times = np.asarray(story_data.data_times, dtype=float)
        order = np.argsort(times)          # onset order is not guaranteed
        words = [words[i] for i in order]
        times = times[order]

        contexts = []
        for index in range(len(words)):
            window = words[max(0, index - context_words + 1):index + 1]
            text = ' '.join(w for w in window if w)
            contexts.append(text if text else '.')
        rows.append(pd.DataFrame({
            'story_id': story,
            'word_index': np.arange(len(words)),
            'word_time_sec': times,
            'sentence': contexts,
        }))
        blocks.append(bold)

    word_stimuli = StimulusSet(pd.concat(rows, ignore_index=True))
    word_stimuli['stimulus_id'] = [
        f'{row.story_id}_w{row.word_index:05d}'
        for row in word_stimuli.itertuples()]
    word_stimuli.identifier = f'{IDENTIFIER}-words{context_words}'
    word_stimuli.stimulus_paths = {}

    data = np.concatenate(blocks, axis=0).astype(np.float32)
    n_neuroid = data.shape[1]
    tr_story = np.concatenate(
        [[story] * len(tr_times_by_story[story]) for story in source.stories])
    tr_time = np.concatenate([tr_times_by_story[s] for s in source.stories])
    assembly = NeuroidAssembly(
        data,
        dims=('presentation', 'neuroid'),
        coords={
            'stimulus_id': ('presentation',
                            [f'{s}_tr{i:04d}' for s, i in zip(
                                tr_story,
                                np.concatenate([np.arange(len(tr_times_by_story[s]))
                                                for s in source.stories]))]),
            'story_id': ('presentation', tr_story),
            'tr_time_sec': ('presentation', tr_time),
            'neuroid_id': ('neuroid', [f'v{i:05d}' for i in range(n_neuroid)]),
            'vertex_index': ('neuroid', np.arange(n_neuroid)),
            'subject': ('neuroid', [SUBJECT] * n_neuroid),
        },
    )
    assembly.name = IDENTIFIER
    return word_stimuli, assembly, tr_times_by_story

