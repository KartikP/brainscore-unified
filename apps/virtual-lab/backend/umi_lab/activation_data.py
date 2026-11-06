"""Read saved neural assemblies in bounded pages, without loading a model."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

PAGE_SIZE = 2048
MAX_METADATA_BYTES = 32 * 1024 * 1024


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
            "stimulus": coords.get("stimulus_id"), "neuroid": coords.get("neuroid_id")}


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
                    labels: bool = False, channel: str | None = None) -> dict[str, Any]:
    """Return one bounded slice; every recorded neuroid is addressable."""
    import numpy as np

    if any(type(value) is not int or value < 0 for value in (index, presentation, offset)):
        raise ValueError("Array, stimulus and neuron indices must be nonnegative integers")
    if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
        raise ValueError(f"Request 1 to {PAGE_SIZE} neurons per page")
    filename = directory / "result.json"
    stat = filename.stat()
    descriptor = _descriptor(str(filename.resolve()), stat.st_size, stat.st_mtime_ns, index, channel)
    array = np.load(array_path(directory, descriptor["data"]), mmap_mode="r", allow_pickle=False)
    dims = descriptor["dims"]
    if len(dims) != array.ndim or array.dtype.kind not in "fiu":
        raise ValueError("Unsupported activation dimensions or numeric type")
    rows, columns = array.shape[dims.index("presentation")], array.shape[dims.index("neuroid")]
    if presentation >= rows or offset >= columns:
        raise ValueError("Stimulus or neuron index is outside the recording")
    if any(size == 0 for size in array.shape):
        raise ValueError("The activation recording is empty")
    selected: list[Any] = [0] * array.ndim
    selected[dims.index("presentation")] = presentation
    selected[dims.index("neuroid")] = slice(offset, min(offset + limit, columns))
    values = array[tuple(selected)]
    result = {"total_rows": rows, "total_columns": columns, "presentation": presentation,
              "stimulus": _label(directory, descriptor["stimulus"], presentation), "offset": offset,
              "values": [float(value) if np.isfinite(value) else None for value in values],
              "slice": {dim: 0 for dim in dims if dim not in ("presentation", "neuroid")}}
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
