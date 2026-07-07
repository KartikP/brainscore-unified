"""Tests for pre-flight checks wired into the unified scoring pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from brainscore_core.compatibility import CompatibilityError
from brainscore_core.io_catalog import modalities_to_input_channels
from brainscore_core.memory import MemoryError


class _LegacyBenchmark:
    identifier = 'legacy-benchmark'

    def __call__(self, model):
        raise NotImplementedError


class TestPreflightInScore:
    """Verify that brainscore.score calls compatibility and memory checks."""

    def _make_model_and_benchmark(self, model_modalities, bench_required):
        model = MagicMock()
        model.identifier = 'test-model'
        model.supported_modalities = model_modalities
        model.available_modalities = model_modalities
        model.required_modalities = set()
        model.region_layer_map = {}
        model.in_channels = modalities_to_input_channels(model_modalities)
        model.out_channels = set()
        model.required_channels = set()

        benchmark = MagicMock(spec=['identifier', 'required_modalities', '__call__'])
        benchmark.identifier = 'test-bench'
        benchmark.required_modalities = bench_required
        return model, benchmark

    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_incompatible_model_raises_compatibility_error(
            self, mock_load_model, mock_load_benchmark):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'text'},
            bench_required={'vision'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(CompatibilityError, match="does not support modalities required"):
            brainscore.score('test-model', 'test-bench', check_mem=False)

    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_compatible_model_proceeds_to_scoring(
            self, mock_load_model, mock_load_benchmark):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        brainscore.score('test-model', 'test-bench', check_mem=False)
        benchmark.assert_called_once_with(model)

    @patch('brainscore_core.compatibility.check_channel_compatibility')
    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_channel_check_runs_for_compatible_pair(
            self, mock_load_model, mock_load_benchmark, mock_check_channel):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        brainscore.score('test-model', 'test-bench', check_mem=False)

        mock_check_channel.assert_called_once_with(model, benchmark)
        benchmark.assert_called_once_with(model)

    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_channel_mismatch_raises_channel_named_error(
            self, mock_load_model, mock_load_benchmark):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        benchmark.required_input_channels = {'text'}
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(CompatibilityError, match="text"):
            brainscore.score('test-model', 'test-bench', check_mem=False)

        benchmark.assert_not_called()

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_check_mem_false_skips_memory_check(
            self, mock_load_model, mock_load_benchmark, mock_check_memory):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        brainscore.score('test-model', 'test-bench', check_mem=False)
        mock_check_memory.assert_not_called()

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_check_mem_true_calls_memory_check(
            self, mock_load_model, mock_load_benchmark, mock_check_memory):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        brainscore.score('test-model', 'test-bench', check_mem=True)
        mock_check_memory.assert_called_once_with(model, benchmark)

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore.load_benchmark')
    @patch('brainscore.load_model')
    def test_memory_error_propagates(
            self, mock_load_model, mock_load_benchmark, mock_check_memory):
        import brainscore

        model, benchmark = self._make_model_and_benchmark(
            model_modalities={'vision'},
            bench_required={'vision'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark
        mock_check_memory.side_effect = MemoryError("not enough memory")

        with pytest.raises(MemoryError, match="not enough memory"):
            brainscore.score('test-model', 'test-bench', check_mem=True)

    @patch('brainscore_vision.load_benchmark')
    def test_vision_fallback_benchmark_gets_modality_metadata(
            self, mock_load_vision_benchmark):
        import brainscore

        benchmark = _LegacyBenchmark()
        mock_load_vision_benchmark.return_value = benchmark

        loaded = brainscore.load_benchmark('legacy-vision-benchmark')

        assert loaded is benchmark
        assert loaded.required_modalities == {'vision'}

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_vision.load_benchmark')
    def test_language_fallback_benchmark_gets_modality_metadata(
            self, mock_load_vision_benchmark, mock_load_language_benchmark):
        import brainscore

        benchmark = _LegacyBenchmark()
        mock_load_vision_benchmark.side_effect = KeyError
        mock_load_language_benchmark.return_value = benchmark

        loaded = brainscore.load_benchmark('legacy-language-benchmark')

        assert loaded is benchmark
        assert loaded.required_modalities == {'text'}

    @patch('brainscore_vision.load_benchmark')
    def test_fallback_benchmark_keeps_explicit_modality_metadata(
            self, mock_load_vision_benchmark):
        import brainscore

        benchmark = _LegacyBenchmark()
        benchmark.required_modalities = {'audio'}
        mock_load_vision_benchmark.return_value = benchmark

        loaded = brainscore.load_benchmark('declared-legacy-benchmark')

        assert loaded.required_modalities == {'audio'}
