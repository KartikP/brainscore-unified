"""Tests for the matplotlib-only visualization renderers. Each must produce a
non-empty PNG without nilearn/nibabel/network access. The cortical-surface
renderers are exercised separately on a machine with the surface assets.
"""
import os

import numpy as np
import pytest

from brainscore.visualization import (
    normalize_values, parcel_grid_heatmap, network_strip,
    ablation_effect_bar, response_heatmap, before_after_difference,
    composite_selection_map, units_per_layer_bar, selectivity_histogram,
    scaling_curve_single, scaling_curves_grid, normalized_scaling_overlay,
)


def _png_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 1000


class TestNormalize:
    def test_scales_to_unit_interval(self):
        out = normalize_values(np.array([0.0, 5.0, 10.0]))
        assert np.isclose(out[0], 0.0) and np.isclose(out[-1], 1.0)

    def test_preserves_nan(self):
        out = normalize_values(np.array([1.0, np.nan, 3.0]))
        assert np.isnan(out[1])

    def test_constant_input_no_crash(self):
        out = normalize_values(np.array([2.0, 2.0, 2.0]))
        assert np.all(np.isfinite(out))


class TestBrainMapFallbacks:
    def test_parcel_grid_heatmap_writes_png(self, tmp_path):
        vals = np.random.RandomState(0).rand(1000)
        out = parcel_grid_heatmap(vals, title='per-parcel r',
                                  out_png=str(tmp_path / 'grid.png'))
        assert _png_nonempty(out)

    def test_network_strip_writes_png(self, tmp_path):
        names = [f'7Networks_LH_Vis_{i}' for i in range(250)] + \
                [f'7Networks_LH_Default_{i}' for i in range(250)]
        vals = np.random.RandomState(1).rand(500)
        out = network_strip(vals, names, title='by network',
                            out_png=str(tmp_path / 'strip.png'))
        assert _png_nonempty(out)


class TestAblationFigures:
    def test_effect_bar_with_sequences(self, tmp_path):
        conditions = {
            'baseline': [0.93, 0.94, 0.92],
            'lesioned': [0.66, 0.68, 0.65],
            'random': [0.91, 0.90, 0.92],
            'restored': [0.93, 0.93, 0.94],
        }
        out = ablation_effect_bar(conditions, chance=0.5,
                                  title='Yeatman lexical decision',
                                  out_png=str(tmp_path / 'abl.png'))
        assert _png_nonempty(out)

    def test_effect_bar_with_mean_sem_tuples(self, tmp_path):
        conditions = {'baseline': (0.93, 0.01), 'lesioned': (0.66, 0.02)}
        out = ablation_effect_bar(conditions, out_png=str(tmp_path / 'abl2.png'))
        assert _png_nonempty(out)

    def test_before_after_difference(self, tmp_path):
        rng = np.random.RandomState(2)
        intact = rng.rand(20, 30)
        lesioned = intact.copy(); lesioned[:, :10] = 0   # ablate some units
        out = before_after_difference(intact, lesioned,
                                      out_png=str(tmp_path / 'diff.png'))
        assert _png_nonempty(out)


class TestSelectionFigures:
    def test_composite_selection_map(self, tmp_path):
        sel = {'blocks.5': [3, 7, 19], 'blocks.10': list(range(0, 100, 5)),
               'blocks.16': [1, 2, 3, 900]}
        units = {'blocks.5': 1024, 'blocks.10': 1024, 'blocks.16': 1024}
        out = composite_selection_map(sel, units,
                                      out_png=str(tmp_path / 'sel.png'))
        assert _png_nonempty(out)

    def test_units_per_layer_bar_fraction(self, tmp_path):
        sel = {'blocks.5': [3, 7], 'blocks.10': list(range(50))}
        units = {'blocks.5': 1024, 'blocks.10': 1024}
        out = units_per_layer_bar(sel, units_per_layer=units, as_fraction=True,
                                  out_png=str(tmp_path / 'upl.png'))
        assert _png_nonempty(out)

    def test_selectivity_histogram(self, tmp_path):
        rng = np.random.RandomState(3)
        scores = rng.randn(2000)
        selected = scores > 1.5
        out = selectivity_histogram(scores, selected,
                                    out_png=str(tmp_path / 'hist.png'))
        assert _png_nonempty(out)


class TestScalingCurves:
    MODELS = ['random-init', 'CLIP-B32', 'Qwen-3B', 'BLIP-2', 'Qwen-7B']

    def test_single_curve_with_floor(self, tmp_path):
        out = scaling_curve_single(
            self.MODELS, [0.10, 0.37, 0.21, 0.39, 0.45],
            null_floor=0.10, capability='IT encoding (r)',
            out_png=str(tmp_path / 'sc.png'))
        assert _png_nonempty(out)

    def test_grid_of_capabilities(self, tmp_path):
        scores = {
            'IT encoding': [0.10, 0.37, 0.21, 0.39, 0.45],
            'ROAR behavior': [0.50, 0.68, 0.74, 0.79, 0.83],
            'game success': [0.0, np.nan, 0.0, np.nan, 0.4],
        }
        floors = {'IT encoding': 0.10, 'ROAR behavior': 0.50, 'game success': 0.2}
        out = scaling_curves_grid(self.MODELS, scores, null_floors=floors,
                                  title='Capabilities scale with model quality',
                                  out_png=str(tmp_path / 'grid.png'))
        assert _png_nonempty(out)

    def test_normalized_overlay(self, tmp_path):
        scores = {
            'IT encoding': [0.10, 0.37, 0.21, 0.39, 0.45],
            'ROAR behavior': [0.50, 0.68, 0.74, 0.79, 0.83],
        }
        out = normalized_scaling_overlay(self.MODELS, scores,
                                         out_png=str(tmp_path / 'ov.png'))
        assert _png_nonempty(out)
