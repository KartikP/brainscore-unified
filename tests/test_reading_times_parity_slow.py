"""Parity of the reading times of unified ``gpt2-small`` with the old ``gpt2`` on Futrell2018.

Real weights and data (slow tier, cloud). Both registrations load the checkpoint ``gpt2``.
The test pins what it ran (library versions, tokenizer classes, the device of each model)
in ``READING_TIMES_PARITY_OUT`` (default ``reading-times-parity.json``). It compares the
token ids of every prefix, the NaN mask, every value bit, the dtype at the metric, and the
raw and normalized Futrell2018 scores. No model is moved for the comparison.
"""
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_gpt2_small_reading_times_equal_the_old_gpt2_on_futrell2018():
    import torch
    import transformers
    from brainscore import model_registry
    from brainscore_core.legacy_facade import subject_for_benchmark
    from brainscore_core.text import prepare_context
    from brainscore_language import load_benchmark, load_model
    from brainscore_language.artificial_subject import ArtificialSubject

    benchmark = load_benchmark('Futrell2018-pearsonr')
    words = list(benchmark.data['word'].values)
    old = load_model('gpt2')
    new = model_registry['gpt2-small']()
    facade = subject_for_benchmark(new, benchmark)
    new_parts = new._segment_likelihood.__self__   # the shared causal-model parts

    old.start_behavioral_task(ArtificialSubject.Task.reading_times)
    old_values = old.digest_text(words)['behavior']
    facade.start_behavioral_task(ArtificialSubject.Task.reading_times)
    new_values = facade.digest_text(words)['behavior']

    spans = [(len(old.tokenizer(prepare_context(words[:i + 1]))['input_ids']),
              len(new_parts.tokenizer(prepare_context(words[:i + 1]))['input_ids']))
             for i in range(len(words))]
    differing = int(np.sum(~((old_values.values == new_values.values)
                             | (np.isnan(old_values.values) & np.isnan(new_values.values)))))
    old_score = benchmark(old)
    new_score = benchmark(subject_for_benchmark(new, benchmark))
    record = {
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'tokenizers': [type(old.tokenizer).__name__, type(new_parts.tokenizer).__name__],
        'devices': [str(next(old.basemodel.parameters()).device),
                    str(next(new_parts.model.parameters()).device)],
        'parts': len(words), 'tokens': spans[-1], 'window': 1024,
        'span_differences': sum(a != b for a, b in spans), 'value_differences': differing,
        'dtypes': [str(old_values.values.dtype), str(new_values.values.dtype)],
        'raw': [float(old_score.raw), float(new_score.raw)],
        'score': [float(old_score), float(new_score)],
        'max_abs_difference': float(np.nanmax(np.abs(old_values.values - new_values.values))),
    }
    with open(os.environ.get('READING_TIMES_PARITY_OUT', 'reading-times-parity.json'), 'w') as f:
        json.dump(record, f, indent=1)
    assert spans[-1][0] > 1024                           # the passage crosses the window
    assert record['span_differences'] == 0
    assert old_values.values.dtype == new_values.values.dtype
    assert record['value_differences'] == 0
    assert record['raw'][0] == record['raw'][1] and record['score'][0] == record['score'][1]
