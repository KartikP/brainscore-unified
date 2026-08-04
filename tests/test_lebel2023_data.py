"""Tests for the LeBel2023 stimulus construction.

``_context_sentences`` decides what text the model is actually shown at each
BOLD sample, so an error here silently changes every score on this benchmark
without failing anything. It is a pure function of words, onset times and TR
times, and is tested here without the source pickle.
"""

import numpy as np
import pytest

from brainscore.data.lebel2023.data import TR_SEC, TRIM_HEAD, TRIM_TAIL, _context_sentences

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


def test_trim_constants_match_the_huth_convention():
    """brain_data[i] is the volume at tr_times[i + TRIM_HEAD].

    The loader asserts len(tr_times) == n_tr + TRIM_HEAD + TRIM_TAIL per story;
    measured on the source, that difference is exactly 15 for all 25 stories.
    """
    assert (TRIM_HEAD, TRIM_TAIL) == (5, 10)
    assert TRIM_HEAD + TRIM_TAIL == 15
    assert TR_SEC == 2.0
