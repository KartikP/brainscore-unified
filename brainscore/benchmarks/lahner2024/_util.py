"""Shared helpers for the Lahner2024 benchmark variants."""


def read_stimulus_ids(assembly):
    """Read stimulus ids from an assembly's presentation axis, whether they
    live as a MultiIndex level or a plain coord, and whether the assembly came
    from native-video extraction (uses 'stimulus_id') or frame-aggregation via
    ``temporal_bin`` (uses 'clip_id').

    Tried in order: MultiIndex level 'stimulus_id', MultiIndex level 'clip_id',
    then a top-level 'stimulus_id'/'clip_id' coord.
    """
    if 'presentation' in assembly.indexes:
        idx = assembly.indexes['presentation']
        if hasattr(idx, 'get_level_values'):
            names = list(idx.names) if hasattr(idx, 'names') else []
            if 'stimulus_id' in names:
                return list(idx.get_level_values('stimulus_id'))
            if 'clip_id' in names:
                return list(idx.get_level_values('clip_id'))
    for col in ('stimulus_id', 'clip_id'):
        if col in assembly.coords:
            return list(assembly[col].values)
    raise KeyError(
        f"assembly has neither 'stimulus_id' nor 'clip_id' on its "
        f"presentation axis; coords={list(assembly.coords)}")
