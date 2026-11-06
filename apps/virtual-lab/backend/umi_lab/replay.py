"""Saved stimulus playback metadata; no subject imports or inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .activation_data import MAX_METADATA_BYTES, _recording, _time_bins, array_path
from .observations import coordinate, scalar

MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_REPLAY_BINS = 65536
MEDIA_TYPES = {
    "video": {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"},
    "audio": {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac"},
    "image": {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"},
}


def _read_json(filename: Path, limit: int) -> Any:
    if filename.stat().st_size > limit:
        raise ValueError("Saved metadata exceeds the replay memory budget; use the artifact export")
    return json.loads(filename.read_text())


def _request(directory: Path) -> dict:
    saved = _read_json(directory / "plan.json", MAX_PLAN_BYTES)
    request = saved.get("request", {})
    if request.get("mode") not in ("protocol", "stimuli") or not request.get("rows"):
        raise ValueError("Replay requires saved custom stimulus rows; this run has no replayable stimulus table")
    return request


def replay_media(directory: Path, *, stimulus: int, kind: str) -> tuple[Path, str]:
    """Expose only a declared media file copied into this run's inputs."""
    request = _request(directory)
    if type(stimulus) is not int or not 0 <= stimulus < len(request["rows"]) or kind not in MEDIA_TYPES:
        raise ValueError("Choose a saved stimulus and a supported media kind")
    return _media_path(directory, request["rows"][stimulus], kind)


def _media_path(directory: Path, row: dict, kind: str) -> tuple[Path, str]:
    name = row.get(f"{kind}_path")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("This stimulus has no saved media file of that kind")
    root = (directory / "inputs").resolve()
    filename = (root / name).resolve()
    mime = MEDIA_TYPES[kind].get(filename.suffix.lower())
    if not mime or not filename.is_relative_to(root) or not filename.is_file():
        raise ValueError("The saved stimulus media is unavailable or unsupported by replay")
    return filename, mime


def _stimulus_ids(directory: Path, assembly: dict, count: int) -> list[str]:
    import numpy as np

    coord, level = coordinate(assembly, "stimulus_id", "presentation")
    if coord is None or coord["dims"] != ["presentation"]:
        raise ValueError("Saved neural output has no stimulus IDs; exact stimulus matching is unavailable")
    data = coord["data"]
    if "$array" in data:
        values = np.load(array_path(directory, data), mmap_mode="r", allow_pickle=False)
    elif "$object_array" in data:
        values = data["$object_array"]
    else:
        raise ValueError("Unsupported recorded stimulus IDs")
    if len(values) != count:
        raise ValueError("Saved stimulus IDs do not match the neural presentation count")
    ids = []
    for value in values:
        value = scalar(value)
        ids.append(str(value[level] if level is not None else value))
    if len(set(ids)) != len(ids):
        raise ValueError("Replay requires distinct saved stimulus IDs")
    return ids


def replay_manifest(directory: Path, *, condition: int = 0, repetition: int = 0) -> dict:
    """Describe all recorded neural channels and their independent native clocks."""
    request = _request(directory)
    protocol = request.get("protocol") or {}
    if request["mode"] == "protocol":
        from .protocol import ProtocolSpec
        spec = ProtocolSpec.parse(protocol)
        conditions = [item["name"] for item in spec.conditions]
        repetitions = spec.repetitions
    else:
        conditions = ["baseline", "intervention", "restored"] if request.get("intervention") else ["baseline"]
        repetitions = request.get("repeats", 1)
    if (type(condition) is not int or not 0 <= condition < len(conditions)
            or type(repetition) is not int or not 0 <= repetition < repetitions):
        raise ValueError("Choose a saved condition and repetition")
    index = condition * repetitions + repetition
    experiment = directory / "experiment"
    results = _read_json(experiment / "result.json", MAX_METADATA_BYTES)
    if not isinstance(results, list) or index >= len(results):
        raise ValueError("This condition and repetition have no saved result")
    saved = results[index]
    outputs = saved["$mapping"] if isinstance(saved, dict) and "$mapping" in saved else [(request.get("output", ""), saved)]
    channels, total_bins = [], 0
    rows = request["rows"]
    ids = [str(row.get("stimulus_id", f"stimulus-{position}")) for position, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError("Replay requires distinct stimulus IDs in the saved plan")
    for name, encoded in outputs:
        if not isinstance(name, str) or not name.startswith("neural:"):
            continue
        if len(channels) >= 16:
            raise ValueError("Replay supports at most 16 recorded neural regions")
        descriptor, array = _recording(experiment, index, name)
        dims = descriptor["dims"]
        count = array.shape[dims.index("time_bin")] if "time_bin" in dims else 0
        total_bins += count
        if total_bins > MAX_REPLAY_BINS:
            raise ValueError("The saved clocks exceed the replay budget; use the Results temporal viewer")
        positions = {value: position for position, value in enumerate(_stimulus_ids(experiment, encoded, array.shape[dims.index("presentation")]))}
        channels.append({"channel": name, "units": array.shape[dims.index("neuroid")],
                         "time_bins": _time_bins(experiment, descriptor, range(count)),
                         "presentations": [positions.get(value) for value in ids],
                         "slice": {dim: 0 for dim in dims if dim not in ("presentation", "neuroid", "time_bin")}})
    if not channels:
        raise ValueError("This run has no saved neural regions; record neural outputs to use Replay")
    stimuli = []
    for position, (row, stimulus_id) in enumerate(zip(rows, ids)):
        media = []
        for kind in MEDIA_TYPES:
            if f"{kind}_path" not in row:
                continue
            try:
                _media_path(directory, row, kind)
                media.append({"kind": kind, "url": f"/api/runs/{directory.name}/media?stimulus={position}&kind={kind}"})
            except ValueError:
                media.append({"kind": kind, "error": "Saved media unavailable or unsupported"})
        stimuli.append({"id": stimulus_id, "text": str(row.get("sentence", row.get("text", ""))),
                        "media": media, "audio_preparation": str(row.get("audio_preparation", ""))})
    return {"index": index, "condition": condition, "repetition": repetition,
            "conditions": conditions, "repetitions": repetitions, "stimuli": stimuli, "channels": channels}


def byte_range(header: str | None, size: int) -> tuple[int, int, int]:
    """One HTTP byte range, with inclusive bounds; reject unsatisfied ranges."""
    import re

    if header is None:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size == 0:
        raise ValueError("Unsupported or unsatisfied media byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Unsatisfied media byte range")
        return max(0, size - length), size - 1, 206
    start, stop = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > stop:
        raise ValueError("Unsatisfied media byte range")
    return start, stop, 206
