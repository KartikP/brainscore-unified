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
    layer_modality_heatmap,
)
from brainscore.visualization.brain_map import parcels_to_vertices


def _png_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 1000


class TestParcelToVertexIndexing:
    """The 1-indexed annot ↔ 0-indexed parcel mapping is the most error-prone
    line in the surface path; test it without needing nilearn/network."""

    def test_correct_one_off_mapping(self):
        # annot labels: vertex 0 = medial wall (0), then parcels 1,2,3
        labels = np.array([0, 1, 1, 2, 3, 3, 3])
        values_h = np.array([10.0, 20.0, 30.0])   # parcels 0,1,2 -> labels 1,2,3
        vtx = parcels_to_vertices(labels, values_h, per_hemi=3)
        assert np.isnan(vtx[0])                    # medial wall stays NaN
        assert vtx[1] == 10.0 and vtx[2] == 10.0   # label 1 -> parcel 0
        assert vtx[3] == 20.0                      # label 2 -> parcel 1
        assert vtx[4] == 30.0 and vtx[6] == 30.0   # label 3 -> parcel 2

    def test_unlabeled_vertices_are_nan(self):
        labels = np.array([0, 0, 1])
        vtx = parcels_to_vertices(labels, np.array([5.0]), per_hemi=1)
        assert np.isnan(vtx[0]) and np.isnan(vtx[1]) and vtx[2] == 5.0


class TestQuickbrainOptional:
    """quickbrain is optional — absence must raise a clear, actionable error
    (with install instructions), never an opaque ImportError or AttributeError."""

    def test_missing_quickbrain_raises_with_install_hint(self):
        import builtins
        import numpy as np
        from brainscore.visualization import quickbrain_outline_map
        real_import = builtins.__import__

        def blocked(name, *a, **k):
            if name == 'quickbrain':
                raise ImportError("No module named 'quickbrain'")
            return real_import(name, *a, **k)

        builtins.__import__ = blocked
        try:
            with pytest.raises(ImportError, match="optional dependency"):
                quickbrain_outline_map(np.zeros(1000))
        finally:
            builtins.__import__ = real_import


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

    def test_layer_modality_heatmap_unequal_rows(self, tmp_path):
        # towers of different depth must share one axis (NaN-padded)
        contrib = {
            'vision': list(np.linspace(0.2, 0.5, 12)),
            'audio': list(np.linspace(0.1, 0.3, 12)),
            'text': list(np.linspace(0.1, 0.25, 6)),
        }
        out = layer_modality_heatmap(contrib, normalize='row',
                                     title='layer contribution per modality',
                                     out_png=str(tmp_path / 'lc.png'))
        assert _png_nonempty(out)


class TestScalingOverlay:
    MODELS = ['random-init', 'CLIP-B32', 'Qwen-3B', 'BLIP-2', 'Qwen-7B']

    def test_normalized_overlay(self, tmp_path):
        scores = {
            'IT encoding': [0.10, 0.37, 0.21, 0.39, 0.45],
            'ROAR behavior': [0.50, 0.68, 0.74, 0.79, 0.83],
        }
        out = normalized_scaling_overlay(self.MODELS, scores,
                                         out_png=str(tmp_path / 'ov.png'))
        assert _png_nonempty(out)


class TestCorticalSurfaceMovie:
    """Orchestration of the temporal renderer — verified WITHOUT nilearn by
    mocking the per-frame cortical_surface_map (the heavy surface render itself
    is exercised separately on a machine with the fsaverage assets)."""

    def _patch(self, monkeypatch):
        import brainscore.visualization.brain_map as bm
        calls = []

        def fake(parcel_values, *, out_png=None, vmin=None, vmax=None, title=None, **kw):
            calls.append({'vmin': vmin, 'vmax': vmax, 'title': title,
                          'out_png': out_png, 'mean': float(np.nanmean(parcel_values))})
            if out_png:
                with open(out_png, 'w') as f:
                    f.write('x')
            return out_png if out_png else 'fig'
        monkeypatch.setattr(bm, 'cortical_surface_map', fake)
        return bm, calls

    def test_one_frame_per_timepoint_with_paths(self, monkeypatch, tmp_path):
        bm, calls = self._patch(monkeypatch)
        vals = np.random.RandomState(0).rand(7, 1000)
        out = bm.cortical_surface_movie(vals, out_dir=str(tmp_path), prefix='bold')
        assert len(out) == 7 and len(calls) == 7
        assert out[0].endswith('bold_000.png') and out[6].endswith('bold_006.png')
        assert all(os.path.exists(p) for p in out)

    def test_shared_scale_is_constant_across_frames(self, monkeypatch, tmp_path):
        bm, calls = self._patch(monkeypatch)
        vals = np.random.RandomState(1).rand(5, 1000) * np.arange(1, 6)[:, None]
        bm.cortical_surface_movie(vals, out_dir=str(tmp_path))
        vmins = {c['vmin'] for c in calls}; vmaxs = {c['vmax'] for c in calls}
        assert len(vmins) == 1 and len(vmaxs) == 1          # one fixed scale for the whole clip
        assert next(iter(vmaxs)) > next(iter(vmins))

    def test_times_drive_titles(self, monkeypatch, tmp_path):
        bm, calls = self._patch(monkeypatch)
        bm.cortical_surface_movie(np.zeros((3, 1000)), times=[0.0, 1.5, 3.0],
                                  out_dir=str(tmp_path), title_fmt='t = {t:.1f}s')
        assert [c['title'] for c in calls] == ['t = 0.0s', 't = 1.5s', 't = 3.0s']

    def test_bad_shape_raises(self, monkeypatch, tmp_path):
        bm, _ = self._patch(monkeypatch)
        with pytest.raises(ValueError, match='2-D'):
            bm.cortical_surface_movie(np.zeros(1000), out_dir=str(tmp_path))
        with pytest.raises(ValueError, match='parcels'):
            bm.cortical_surface_movie(np.zeros((3, 17)), out_dir=str(tmp_path))
