"""Read saved neural assemblies in bounded pages, without loading a model."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

PAGE_SIZE = 2048
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_TEMPORAL_CELLS = 65536
MAX_TIME_BINS = 1024


def array_path(directory: Path, reference: dict) -> Path:
    digest = reference.get("$array", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Unsupported or invalid recorded array reference")
    path = (directory / "arrays" / f"{digest}.npy").resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Recorded activation array is unavailable")
    return path


@lru_cache(maxsize=4)
def _descriptor(filename: str, size: int, modified_ns: int, index: int, channel: str | None = None) -> dict:
    if size > MAX_METADATA_BYTES:
        raise ValueError("Recording metadata exceeds the viewer's memory budget; use the exported artifacts")
    with Path(filename).open() as stream:
        results = json.load(stream)
    if not isinstance(results, list) or not 0 <= index < len(results):
        raise ValueError("Unknown recorded output index")
    assembly = results[index]
    if isinstance(assembly, dict) and "$mapping" in assembly:
        matching = [value for name, value in assembly["$mapping"] if name == channel]
        if len(matching) != 1:
            raise ValueError("Choose a recorded output channel")
        assembly = matching[0]
    if not isinstance(assembly, dict) or "$assembly" not in assembly or not {"presentation", "neuroid"}.issubset(assembly.get("dims", [])):
        raise ValueError("This output is not a recorded neural assembly")
    coords = assembly.get("coords", {})
    return {"dims": assembly["dims"], "data": assembly["data"],
            "stimulus": coords.get("stimulus_id"), "neuroid": coords.get("neuroid_id"),
            "assembly": assembly}


def _recording(directory: Path, index: int, channel: str | None) -> tuple[dict, Any]:
    import numpy as np

    if type(index) is not int or index < 0:
        raise ValueError("The output index must be a nonnegative integer")
    filename = directory / "result.json"
    stat = filename.stat()
    descriptor = _descriptor(str(filename.resolve()), stat.st_size, stat.st_mtime_ns, index, channel)
    array = np.load(array_path(directory, descriptor["data"]), mmap_mode="r", allow_pickle=False)
    if len(descriptor["dims"]) != array.ndim or array.dtype.kind not in "fiu":
        raise ValueError("Unsupported activation dimensions or numeric type")
    if any(size == 0 for size in array.shape):
        raise ValueError("The activation recording is empty")
    return descriptor, array


def _time_bins(directory: Path, descriptor: dict, indices: range) -> list[dict]:
    """Read recorded bounds, including old xarray MultiIndex encodings."""
    import numpy as np
    from .observations import coordinate, scalar

    result = [{"index": index, "start_ms": None, "end_ms": None, "center_ms": None} for index in indices]
    for key, name in (("start_ms", "time_bin_start_ms"), ("end_ms", "time_bin_end_ms"),
                      ("center_ms", "time_bin_center_ms")):
        coord, level = coordinate(descriptor["assembly"], name, "time_bin")
        if coord is None or coord["dims"] != ["time_bin"]:
            continue
        data = coord["data"]
        if "$array" in data:
            values = np.load(array_path(directory, data), mmap_mode="r", allow_pickle=False)
        elif "$object_array" in data:
            values = data["$object_array"]
        else:
            continue
        for row, index in zip(result, indices):
            value = scalar(values[index])
            if level is not None:
                value = value[level]
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                row[key] = float(value)
    return result


def _label(directory: Path, coordinate: dict | None, index: int) -> str:
    if coordinate is None:
        return str(index)
    data = coordinate["data"]
    if isinstance(data, dict) and "$array" in data:
        import numpy as np
        array = np.load(array_path(directory, data), mmap_mode="r", allow_pickle=False)
        return str(array[index])[:160]
    if isinstance(data, dict) and "$object_array" in data:
        value = data["$object_array"][index]
        return str(value)[:160]
    return str(index)


def activation_page(directory: Path, *, index: int = 0, presentation: int = 0,
                    offset: int = 0, limit: int = PAGE_SIZE, stats: bool = False,
                    labels: bool = False, channel: str | None = None,
                    time_bin: int = 0) -> dict[str, Any]:
    """Return one bounded slice; every recorded neuroid is addressable."""
    import numpy as np

    if any(type(value) is not int or value < 0 for value in (index, presentation, offset, time_bin)):
        raise ValueError("Array, stimulus and neuron indices must be nonnegative integers")
    if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
        raise ValueError(f"Request 1 to {PAGE_SIZE} neurons per page")
    descriptor, array = _recording(directory, index, channel)
    dims = descriptor["dims"]
    rows, columns = array.shape[dims.index("presentation")], array.shape[dims.index("neuroid")]
    if presentation >= rows or offset >= columns:
        raise ValueError("Stimulus or neuron index is outside the recording")
    selected: list[Any] = [0] * array.ndim
    count = array.shape[dims.index("time_bin")] if "time_bin" in dims else 1
    if time_bin >= count:
        raise ValueError("Time bin is outside the recording")
    if "time_bin" in dims:
        selected[dims.index("time_bin")] = time_bin
    selected[dims.index("presentation")] = presentation
    selected[dims.index("neuroid")] = slice(offset, min(offset + limit, columns))
    values = array[tuple(selected)]
    result = {"total_rows": rows, "total_columns": columns, "presentation": presentation,
              "stimulus": _label(directory, descriptor["stimulus"], presentation), "offset": offset,
              "values": [float(value) if np.isfinite(value) else None for value in values],
              "slice": {dim: 0 for dim in dims if dim not in ("presentation", "neuroid")}}
    if "time_bin" in dims:
        result["slice"]["time_bin"] = time_bin
        result["time"] = {"count": count, **_time_bins(directory, descriptor, range(time_bin, time_bin + 1))[0]}
    if labels:
        result["labels"] = [_label(directory, descriptor["neuroid"], neuron)
                            for neuron in range(offset, min(offset + limit, columns))]
    if stats:
        extent = 0.0
        for start in range(0, columns, PAGE_SIZE):
            selected[dims.index("neuroid")] = slice(start, min(start + PAGE_SIZE, columns))
            chunk = array[tuple(selected)]
            finite = chunk[np.isfinite(chunk)]
            if finite.size:
                extent = max(extent, float(np.max(np.abs(finite))))
        result["extent"] = extent or 1.0
    return result


def temporal_page(directory: Path, *, index: int = 0, presentation: int = 0,
                  offset: int = 0, limit: int = 256, time_offset: int = 0,
                  time_limit: int = 128, neurons: list[int] | None = None,
                  channel: str | None = None, labels: bool = True) -> dict[str, Any]:
    """Read a bounded time-by-neuron window from saved arrays only."""
    import numpy as np

    if any(type(value) is not int or value < 0 for value in (presentation, offset, time_offset)):
        raise ValueError("Stimulus, neuron and time offsets must be nonnegative integers")
    if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
        raise ValueError(f"Request 1 to {PAGE_SIZE} neurons per page")
    if type(time_limit) is not int or not 1 <= time_limit <= MAX_TIME_BINS:
        raise ValueError(f"Request 1 to {MAX_TIME_BINS} time bins per page")
    descriptor, array = _recording(directory, index, channel)
    dims = descriptor["dims"]
    if "time_bin" not in dims:
        raise ValueError("This neural output has no recorded time axis")
    rows, columns, times = (array.shape[dims.index(dim)] for dim in ("presentation", "neuroid", "time_bin"))
    if presentation >= rows or offset >= columns or time_offset >= times:
        raise ValueError("Stimulus, neuron or time offset is outside the recording")
    if neurons is not None and (not isinstance(neurons, list) or not 1 <= len(neurons) <= 8
            or any(type(neuron) is not int or not 0 <= neuron < columns for neuron in neurons)
            or len(set(neurons)) != len(neurons)):
        raise ValueError("Choose 1 to 8 distinct recorded neuron indices")
    units = neurons if neurons is not None else list(range(offset, min(offset + limit, columns)))
    stop = min(time_offset + time_limit, times)
    if len(units) * (stop - time_offset) > MAX_TEMPORAL_CELLS:
        raise ValueError(f"The temporal window exceeds {MAX_TEMPORAL_CELLS} values; reduce its range")
    selection: list[Any] = [0] * array.ndim
    selection[dims.index("presentation")] = presentation
    selection[dims.index("time_bin")] = slice(time_offset, stop)
    # Restrict the neuron axis before materializing a window. No full array load.
    selection[dims.index("neuroid")] = slice(None) if neurons is not None else slice(offset, offset + len(units))
    window = array[tuple(selection)]
    remaining = [dim for dim in dims if dim in ("time_bin", "neuroid")]
    if remaining != ["time_bin", "neuroid"]:
        window = window.T
    if neurons is not None:
        window = window[:, units]
    finite = window[np.isfinite(window)]
    return {"presentation": presentation, "stimulus": _label(directory, descriptor["stimulus"], presentation),
            "total_rows": rows, "total_columns": columns, "total_time_bins": times,
            "offset": offset, "time_offset": time_offset, "neurons": units,
            "labels": [_label(directory, descriptor["neuroid"], neuron) for neuron in units] if labels else [],
            "time_bins": _time_bins(directory, descriptor, range(time_offset, stop)),
            "values": [[float(value) if np.isfinite(value) else None for value in row] for row in window],
            "extent": float(np.max(np.abs(finite))) if finite.size and np.any(finite) else 1.0,
            "slice": {dim: 0 for dim in dims if dim not in ("presentation", "neuroid", "time_bin")}}
