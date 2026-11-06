"""Scientific display checks: bounded, labeled, raw values with honest slices."""
import json

import numpy as np
import xarray as xr

from umi_lab.previews import PREVIEW_COLUMNS, PREVIEW_ROWS, activation_preview
from umi_lab.workbench import response_preview
from umi_lab.worker import bounded_message


def test_activation_preview_preserves_orientation_labels_missing_values_and_slice():
    values = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    values[0, 1, 2] = np.nan
    assembly = xr.DataArray(values, dims=("time_bin", "neuroid", "presentation"),
                            coords={"stimulus_id": ("presentation", ["a", "b", "c"]),
                                    "neuroid_id": ("neuroid", ["u0", "u1", "u2", "u3"])})
    before = assembly.copy(deep=True)
    preview = activation_preview(assembly)
    assert preview["slice"] == {"time_bin": 0}
    assert preview["row_labels"] == ["a", "b", "c"]
    assert preview["column_labels"] == ["u0", "u1", "u2", "u3"]
    assert preview["values"][1][2] == values[0, 2, 1]
    assert preview["values"][2][1] is None
    xr.testing.assert_identical(assembly, before)
    json.dumps(preview, allow_nan=False)


def test_preview_is_a_bounded_top_left_window_not_an_average_or_normalization():
    values = np.arange(20 * 80, dtype=float).reshape(20, 80) - 900
    assembly = xr.DataArray(values, dims=("presentation", "neuroid"))
    preview = response_preview(assembly)["activation_preview"]
    assert len(preview["values"]) == PREVIEW_ROWS
    assert len(preview["values"][0]) == PREVIEW_COLUMNS
    assert preview["total_rows"] == 20 and preview["total_columns"] == 80
    np.testing.assert_array_equal(preview["values"], values[:PREVIEW_ROWS, :PREVIEW_COLUMNS])
    bounded_message({"outputs": [preview] * 4})


def test_empty_non_numeric_and_unrelated_assemblies_do_not_invent_activations():
    assert activation_preview(xr.DataArray(np.zeros((0, 3)), dims=("presentation", "neuroid"))) is None
    assert activation_preview(xr.DataArray([["x"]], dims=("presentation", "neuroid"))) is None
    assert activation_preview(xr.DataArray([[1]], dims=("presentation", "choice"))) is None


def test_calibration_probe_plots_real_recording_events_and_restores(tmp_path):
    from umi_lab.demo import create_profile
    from umi_lab.experiment import Plan, execute

    result = execute(create_profile(), Plan(activity=True, ablation=True), tmp_path / "run")
    previews = {preview["condition"]: preview for preview in result["report"]["activation_previews"]}
    assert set(previews) == {"baseline", "ablation", "random_control", "restored"}
    baseline, changed, restored = (previews[name] for name in ("baseline", "ablation", "restored"))
    assert baseline["row_axis"] == "Capture"
    assert baseline["values"][0][:2] == [4.0, 0.0]
    assert changed["values"][0][:2] == [0.0, 0.0]
    assert baseline["values"] == restored["values"]
    assert len(baseline["values"]) <= PREVIEW_ROWS
    bounded_message(result)
