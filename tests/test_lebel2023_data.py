"""Tests for the LeBel2023 stimulus construction.

``_context_sentences`` decides what text the model is actually shown at each
BOLD sample, so an error here silently changes every score on this benchmark
without failing anything. It is a pure function of words, onset times and TR
times, and is tested here without the source pickle.
"""

import numpy as np
import pytest

from brainscore.data.lebel2023.data import (
    TR_SEC, TRIM_HEAD, TRIM_TAIL, _context_sentences)

pytestmark = pytest.mark.unit


class TestContextSentences:
    def test_window_holds_only_words_spoken_before_the_sample(self):
        words = ['a', 'b', 'c', 'd']
        times = [0.5, 1.5, 2.5, 3.5]
        # sample at t=2.0 with a 10 s window: 'a' and 'b' have been heard, not 'c'
        assert _context_sentences(words, times, np.array([2.0]), 10.0) == ['a b']

    def test_window_start_is_exclusive_of_older_words(self):
        """A word that fell out of the window must not reappear."""
        words = ['old', 'recent']
        times = [0.0, 9.0]
        # 4 s window ending at t=10 excludes the word spoken at t=0
        assert _context_sentences(words, times, np.array([10.0]), 4.0) == ['recent']

    def test_consecutive_samples_overlap(self):
        """Overlap is intended: it is the rolling context a listener has."""
        words = list('abcdef')
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        got = _context_sentences(words, times, np.array([3.0, 4.0]), 3.0)
        assert got == ['a b c d', 'b c d e']

    def test_silent_window_yields_a_non_empty_string(self):
        """Tokenizers must never be handed an empty string."""
        got = _context_sentences(['x'], [0.0], np.array([100.0]), 5.0)
        assert got == ['.']
        assert all(s.strip() for s in got)

    def test_unordered_word_times_are_sorted_first(self):
        """The source does not guarantee onset order."""
        words = ['second', 'first']
        times = [2.0, 1.0]
        assert _context_sentences(words, times, np.array([3.0]), 10.0) == ['first second']

    def test_blank_words_are_dropped_not_rendered(self):
        """The source's first entry is an empty string; it must not add spaces."""
        got = _context_sentences(['', 'hello', '  ', 'world'],
                                 [0.0, 1.0, 2.0, 3.0], np.array([4.0]), 10.0)
        assert got == ['hello world']

    def test_one_sentence_per_sample(self):
        tr_times = np.arange(0.0, 20.0, TR_SEC)
        got = _context_sentences(list('abcde'), [1.0, 3.0, 5.0, 7.0, 9.0], tr_times, 10.0)
        assert len(got) == len(tr_times)


def test_sampling_interval():
    assert TR_SEC == 2.0


class TestTrimDirection:
    """The 15 spare ``tr_times`` split 10 from the head and 5 from the tail.

    Getting this backwards is not caught by any consistency check: 5/10 and 10/5
    both satisfy ``len(tr_times) == n_tr + 15``, which is what the loader asserts.
    The wrong direction places features five samples early, so the regression
    fits BOLD from words up to six seconds in the *future* — physiologically
    impossible, and it roughly halved every score on this benchmark before it was
    found. The reference pipeline hardcodes ``downsampled[10:-5]``.
    """

    def test_head_is_larger_than_tail(self):
        assert TRIM_HEAD == 10 and TRIM_TAIL == 5
        assert TRIM_HEAD > TRIM_TAIL, (
            'inverting the trim misaligns features against BOLD by five samples')

    def test_total_is_unchanged(self):
        """Both directions satisfy this — which is exactly why it is not enough."""
        assert TRIM_HEAD + TRIM_TAIL == 15

    def test_build_indexes_tr_times_from_the_head_offset(self, monkeypatch, tmp_path):
        """The emitted TR times must start at ``tr_times[TRIM_HEAD]``.

        Pins the indexing itself, not just the constants, so a change to either
        one without the other fails here.
        """
        import numpy as np
        import brainscore.data.lebel2023.data as data_module

        n_tr = 12
        all_tr_times = np.arange(-9.0, -9.0 + 2.0 * (n_tr + 15), 2.0)

        class _Story:
            brain_data = np.zeros((n_tr, 3), dtype=np.float32)
            tr_times = all_tr_times
            words = ['a', 'b', 'c']
            data_times = np.array([0.5, 1.5, 2.5])

        class _Source:
            stories = ['only']
            story_data = {'only': _Story()}

        probe = tmp_path / 'fake.pkl'
        probe.write_bytes(b'not really a pickle')
        monkeypatch.setattr(data_module, '_read_pickle', lambda path: _Source())

        stimuli, _ = data_module.build(pickle_path=probe)
        emitted = np.asarray(stimuli['tr_time_sec'].values)
        assert np.allclose(emitted, all_tr_times[TRIM_HEAD:TRIM_HEAD + n_tr])
        # and explicitly not the inverted convention
        assert not np.allclose(emitted, all_tr_times[TRIM_TAIL:TRIM_TAIL + n_tr])


def test_cache_key_includes_the_trim(tmp_path, monkeypatch):
    """A cache written under a different trim must not be reused.

    The trim was inverted once. A file written under the old convention looks
    identical on inspection and simply scores half as well, so the cache has to
    treat it as a different artifact rather than a hit.
    """
    import brainscore.data.lebel2023.data as data_module
    monkeypatch.setattr(data_module, '_cache_dir', lambda: tmp_path)

    seen = []
    monkeypatch.setattr(data_module, 'build',
                        lambda **kw: seen.append(kw) or (_ for _ in ()).throw(
                            RuntimeError('stop after cache miss')))

    for head, tail in ((10, 5), (5, 10)):
        monkeypatch.setattr(data_module, 'TRIM_HEAD', head)
        monkeypatch.setattr(data_module, 'TRIM_TAIL', tail)
        # touch a cache file for this trim, then confirm the other trim misses it
        for existing in tmp_path.glob('*.nc'):
            existing.unlink()
        (tmp_path / f'lebel_uts03_context5tr_trim{head}-{tail}.nc').touch()
        monkeypatch.setattr(data_module, 'TRIM_HEAD', tail)
        monkeypatch.setattr(data_module, 'TRIM_TAIL', head)
        try:
            data_module.load()
        except RuntimeError as error:
            assert 'stop after cache miss' in str(error)
        else:
            raise AssertionError(
                f'cache written at {head}/{tail} was reused at {tail}/{head}')
