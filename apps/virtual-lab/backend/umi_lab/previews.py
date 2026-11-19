"""Fixed-size numerical previews; full recordings remain in run artifacts."""
from __future__ import annotations

from typing import Any

PREVIEW_ROWS = 8
PREVIEW_COLUMNS = 24
LABEL_LENGTH = 80


def finite_values(values: Any) -> list[float | None]:
    import numpy as np

    return [float(value) if np.isfinite(value) else None for value in values]


def activation_preview(assembly: Any) -> dict | None:
    """Slice before conversion, without pooling or copying the full assembly."""
    import numpy as np

    if not {"presentation", "neuroid"}.issubset(assembly.dims):
        return None
    if assembly.dtype.kind not in "fiu" or any(size == 0 for size in assembly.sizes.values()):
        return None
    extra = {dim: 0 for dim in assembly.dims if dim not in ("presentation", "neuroid")}
    window = assembly.isel({**extra, "presentation": slice(0, PREVIEW_ROWS),
                            "neuroid": slice(0, PREVIEW_COLUMNS)}).transpose("presentation", "neuroid")

    def labels(axis: str, coordinate: str) -> list[str]:
        if coordinate in window.coords and window[coordinate].dims == (axis,):
            return [str(value)[:LABEL_LENGTH] for value in window[coordinate].values]
        return [str(index) for index in range(window.sizes[axis])]

    return {"values": [finite_values(row) for row in np.asarray(window.values)],
            "row_labels": labels("presentation", "stimulus_id"),
            "column_labels": labels("neuroid", "neuroid_id"),
            "total_rows": assembly.sizes["presentation"], "total_columns": assembly.sizes["neuroid"],
            "row_axis": "Stimulus", "column_axis": "Neuroid", "slice": extra}


def activity_preview_tool() -> Any:
    """Observe existing recording events; never install another model hook."""
    from brainscore.experiments import Tool

    class ActivityPreview(Tool):
        name = "lab_activity_preview"

        def __init__(self) -> None:
            self.previews: dict[str, dict] = {}

        def on_event(self, event: dict) -> None:
            if event["kind"] != "activity":
                return
            payload = event["payload"]
            captured = payload["value"]
            array = captured["array"]
            if array.dtype.kind not in "fiu" or array.ndim == 0 or not array.size:
                return
            condition = str(event["condition"])
            preview = self.previews.setdefault(condition, {
                "condition": condition, "target": str(payload["target"])[:LABEL_LENGTH],
                "values": [], "row_labels": [], "column_labels": [], "total_rows": 0,
                "total_columns": 0, "row_axis": "Capture", "column_axis": "Unit",
                "slice": {"other_axes": "index 0"},
            })
            preview["total_rows"] += 1
            if len(preview["values"]) >= PREVIEW_ROWS:
                return
            axis = captured.get("axis")
            axis = (axis if axis is not None else -1) % array.ndim
            selection = [0] * array.ndim
            selection[axis] = slice(0, PREVIEW_COLUMNS)
            preview["values"].append(finite_values(array[tuple(selection)]))
            preview["row_labels"].append(f"Capture {len(preview['values'])}")
            preview["total_columns"] = array.shape[axis]
            preview["column_labels"] = [str(index) for index in range(min(array.shape[axis], PREVIEW_COLUMNS))]

    return ActivityPreview()
