"""Neural encoding capability."""

from .base import Capability
from .registry import register_capability
from brainscore_core.io_catalog import canonical_modality


class NeuralEncodingCapability(Capability):
    """Default StimulusSet -> NeuroidAssembly path."""

    identifier = 'neural-encoding'
    order = 1000

    def enabled_for(self, subject) -> bool:
        return bool(subject._input_keys)

    def handles(self, subject, event, **kwargs) -> bool:
        del kwargs
        try:
            event.columns
        except AttributeError:
            return False
        return True

    def process(self, subject, stimuli, **kwargs):
        multi_modality = kwargs.get('multi_modality', False)
        detected = subject.detect_modalities(stimuli)

        if not detected:
            from brainscore_core.io_catalog import stimulus_columns
            from brainscore_core.streaming import parse_channel
            accepted = {m: sorted(c for c, channel in stimulus_columns().items()
                                  if parse_channel(channel)[0] == canonical_modality(m))
                        for m in subject.supported_modalities}
            raise ValueError(
                f"No column in this stimulus set says what kind of input it holds "
                f"(columns present: {list(stimuli.columns)}). Add one column per input "
                f"this subject takes; each row holds the input itself or its file path, "
                f"for example stimuli['image_path'] = [...] for images. Columns this "
                f"subject reads: {accepted}. A stimulus_paths dict alone is not enough: "
                f"the column names the input type."
            )

        layers = subject._recorder.current_layers()
        modality = subject._dispatcher.pick_modality(detected)
        requested_modalities = {
            subject._region_modality_map[r] for r in subject._recorder.recording_regions
            if r in subject._region_modality_map}
        requested_modalities = {canonical_modality(m) for m in requested_modalities}
        missing = requested_modalities - detected
        if missing:
            raise ValueError(f'Recording targets require missing input modalities: {sorted(missing)}.')
        if len(requested_modalities) == 1:
            modality = next(iter(requested_modalities))
        elif len(requested_modalities) > 1 and not multi_modality:
            raise ValueError('Recording targets span modalities; pass multi_modality=True.')

        # Guard BOTH paths: if regions ARE defined but none is active, the user
        # forgot start_recording (an empty region_layer_map is a legitimate
        # default-extraction path). Check only the modality/modalities that will
        # ACTUALLY run — the selected one for single dispatch, all detected for a
        # multi-modality fan-out — so a non-selected layer-aware tower does not
        # false-positive the selected plain-callable path.
        will_run = detected if (multi_modality and len(detected) > 1) else {modality}
        if (not layers and subject._region_layer_map_dict
                and not subject._recorder.composite_recording
                and any(subject._dispatcher.supports_layer_extraction(m) for m in will_run)):
            raise ValueError(
                f"Subject '{subject.identifier}' records no region: request a "
                f"neural channel in the session, for example "
                f"stimulus_session(stimuli, ['neural:IT']). Without one the "
                f"recording has 0 neuroids. Known regions: "
                f"{sorted(subject._region_layer_map_dict.keys())}."
            )

        from .._model.dispatch import fused_for
        if (multi_modality or fused_for(subject, detected) is not None) and len(detected) > 1:
            return subject._dispatcher.process_multi_modality(stimuli, detected, layers)

        if not multi_modality and len(detected) > 1:
            import warnings
            warnings.warn(
                f"Stimulus set contains multiple supported modalities "
                f"{sorted(detected)} for subject '{subject.identifier}'. "
                f"Default process(..., multi_modality=False) will use "
                f"'{modality}' via recording-target routing or MODALITY_PRIORITY. Pass "
                f"multi_modality=True to extract all supported modalities, "
                f"or provide stimulus columns for only the modality you want.",
                UserWarning,
                stacklevel=2,
            )
        if (subject._recorder.composite_recording
                and subject._dispatcher.supports_layer_extraction(modality)):
            return subject._recorder.process_composite_regions(stimuli, modality)
        layers = subject._recorder.filter_layers_for_modality(layers, modality)
        assembly = subject._dispatcher.extract_for_modality(stimuli, modality, layers)
        if subject._recorder.is_multi_region and subject._dispatcher.supports_layer_extraction(modality):
            assembly = subject._recorder.tag_neuroids_with_regions(assembly, modality)
        return assembly


register_capability(NeuralEncodingCapability())
