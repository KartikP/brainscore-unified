"""Bounded views of saved protocol outputs; never reopen or run a subject."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .activation_data import MAX_METADATA_BYTES, array_path


def scalar(value: Any) -> Any:
    if isinstance(value, dict):
        if "$float" in value:
            return None
        if "$tuple" in value:
            return tuple(scalar(item) for item in value["$tuple"])
        if "$mapping" in value:
            return {str(key): scalar(item) for key, item in value["$mapping"][:32]}
        return None
    if isinstance(value, list):
        return [scalar(item) for item in value[:32]]
    return value


def coordinate(assembly: dict, name: str, axis: str) -> tuple[dict | None, int | None]:
    coords = assembly.get("coords", {})
    if name in coords:
        return coords[name], None
    names = assembly.get("multi_indexes", {}).get(axis, [])
    if name in names and axis in coords:
        return coords[axis], names.index(name)
    return None, None


def coordinate_value(directory: Path, entry: tuple[dict | None, int | None], index: int) -> Any:
    import numpy as np
    coord, level = entry
    if coord is None:
        return None
    data = coord["data"]
    if "$array" in data:
        value = np.load(array_path(directory, data), mmap_mode="r", allow_pickle=False)[index]
        value = value.item() if np.ndim(value) == 0 else value
    elif "$object_array" in data:
        value = scalar(data["$object_array"][index])
    else:
        return None
    return value[level] if level is not None else value


def preview(directory: Path, encoded: dict) -> dict:
    import numpy as np
    import xarray as xr
    from .workbench import response_preview

    dims, data = encoded["dims"], encoded["data"]
    if "$array" in data:
        array = np.load(array_path(directory, data), mmap_mode="r", allow_pickle=False)
        shape = array.shape
        limits = {"presentation": 8 if "neuroid" in dims else 20, "neuroid": 24}
        selection = tuple(slice(0, limits.get(dim, size)) if dim in ("presentation", "neuroid", "choice") else 0
                          for dim, size in zip(dims, shape))
        window = array[selection]
        shown_dims = [dim for dim in dims if dim in ("presentation", "neuroid", "choice")]
    elif "$object_array" in data and dims == ["presentation"]:
        shape = tuple(data["shape"])
        window = np.array([scalar(value) for value in data["$object_array"][:20]], dtype=object)
        shown_dims = dims
    else:
        raise ValueError("This recorded output needs a specialized viewer")
    coords = {}
    for axis, names in (("presentation", ("stimulus_id", "sentence", "answer_valid", "raw_answer", "parse_error")),
                        ("neuroid", ("neuroid_id",)), ("choice", ("choice",))):
        if axis not in shown_dims:
            continue
        count = window.shape[shown_dims.index(axis)]
        for name in names:
            entry = coordinate(encoded, name, axis)
            if entry[0] is not None:
                coords[name] = (axis, [coordinate_value(directory, entry, index) for index in range(count)])
        if axis == "presentation" and "stimulus_id" not in coords:
            coords["stimulus_id"] = (axis, [str(index) for index in range(count)])
    assembly = xr.DataArray(window, dims=shown_dims, coords=coords)
    attrs = scalar(encoded.get("attrs", {"$mapping": []})) or {}
    assembly.attrs = {key: value for key, value in attrs.items() if isinstance(value, (str, int, float, bool))}
    result = response_preview(assembly)
    result.update(type=encoded["$assembly"], shape=list(shape), dims=dims)
    if "activation_preview" in result:
        result["activation_preview"].update(total_rows=shape[dims.index("presentation")],
            total_columns=shape[dims.index("neuroid")], slice={dim: 0 for dim in dims if dim not in shown_dims})
    if shown_dims == ["presentation"]:
        ids = list(assembly["stimulus_id"].values)
        if window.dtype.kind in "fiu":
            result["scalar_rows"] = [{"stimulus": str(identifier), "value": float(value) if np.isfinite(value) else None}
                                     for identifier, value in zip(ids, window)]
        else:
            result.pop("sample", None)
            result.update(count=shape[0], preview=[{
                "stimulus": str(identifier), "raw_answer": str(value) if value is not None else "",
                "valid": bool(assembly["answer_valid"].values[index]) if "answer_valid" in coords else value is not None,
                "parse_error": str(assembly["parse_error"].values[index]) if "parse_error" in coords else None,
            } for index, (identifier, value) in enumerate(zip(ids, window))])
    if "recording" in attrs:
        result["recording"] = attrs["recording"]
    if "measure" in attrs:
        result["measure"] = attrs["measure"]
    return result


def observation_page(directory: Path, *, index: int = 0, channel: str | None = None) -> list[dict]:
    if type(index) is not int or index < 0:
        raise ValueError("The repetition index must be a nonnegative integer")
    filename = directory / "result.json"
    if filename.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Recording metadata exceeds the viewer memory budget; use the complete artifacts")
    results = json.loads(filename.read_text())
    if not isinstance(results, list) or index >= len(results):
        raise ValueError("Unknown recorded repetition")
    saved = results[index]
    if isinstance(saved, dict) and "$mapping" in saved:
        outputs = saved["$mapping"]
    else:
        plan = json.loads((directory / "experiment.json").read_text())["plan"]
        channels = plan["protocol"]["config"]["output_channels"]
        if len(channels) != 1:
            raise ValueError("The recording does not identify a unique output channel")
        outputs = [(channels[0], saved)]
    if channel is not None:
        outputs = [(name, assembly) for name, assembly in outputs if name == channel]
        if not outputs:
            raise ValueError("This output channel was not recorded")
    if len(outputs) > 32:
        raise ValueError("This recording exceeds the interactive channel budget; use the complete artifacts")
    return [{"channel": channel, "result_index": index, "response": preview(directory, assembly)}
            for channel, assembly in outputs]
