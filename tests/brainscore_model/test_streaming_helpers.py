import numpy as np
import pandas as pd
import xarray as xr

from brainscore import BrainScoreModel
from brainscore_core.model_interface import Perturbation, PerturbationApplied, Selection, StateChange, TaskContext
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    apply_state_change,
    environment_session,
    behavioral_response,
    neural_response,
    state_change_session,
    run_environment,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from .test_behavioral_readout import (
    _fake_preprocessor_returning,
    _make_image_stimulus_set,
)
from .test_closed_loop_rows import (
    _droid_row,
    _make_model as _make_environment_model,
)
from .test_model_interface import (
    _XArrayActivationsModel,
    _XArrayPreprocessor,
    make_stub_preprocessor,
)
from .test_state_change import (
    _counting_state_change_fn,
    _make_model as _make_state_change_model,
)


def _metadata_stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        "stimulus_id": ["s0", "s1"],
        "image_path": ["s0.png", "s1.png"],
        "object_name": ["cat", "dog"],
        "difficulty": [1, 2],
    }))
    stimuli.identifier = "metadata-synthetic"
    return stimuli


def _multimodal_stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        "stimulus_id": ["s0", "s1"],
        "image_file_name": ["s0.png", "s1.png"],
        "sentence": ["hello", "world"],
    }))
    stimuli.identifier = "multimodal-synthetic"
    return stimuli


class _MetadataPreservingExtractor:
    identifier = "metadata-preserving-extractor"

    def __init__(self):
        self.last_stimuli = None

    def __call__(self, stimuli, layers=None, **kwargs):
        del kwargs
        self.last_stimuli = stimuli
        layer = (layers or ["layer4"])[0]
        data = np.array([[1.0, 2.0], [3.0, 4.0]])[:len(stimuli)]
        return NeuroidAssembly(
            data,
            coords={
                "stimulus_id": (
                    "presentation", list(stimuli["stimulus_id"].values)
                ),
                "object_name": (
                    "presentation", list(stimuli["object_name"].values)
                ),
                "difficulty": (
                    "presentation", list(stimuli["difficulty"].values)
                ),
                "neuroid_id": (
                    "neuroid", [f"{layer}.{index}" for index in range(data.shape[1])]
                ),
                "layer": ("neuroid", [layer] * data.shape[1]),
            },
            dims=["presentation", "neuroid"],
        )


class _ParticipateTrackingBrainScoreModel(BrainScoreModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.participate_called = False
        self.participate_requested_output_channels = None

    def participate(self, session):
        self.participate_called = True
        self.participate_requested_output_channels = tuple(
            session.requested_output_channels
        )
        return super().participate(session)


def _native_neural_model(extractor, model_cls=BrainScoreModel):
    return model_cls(
        identifier="native-neural",
        model=None,
        region_layer_map={"IT": "layer4"},
        preprocessors={"vision": extractor},
    )


def _native_state_change_model(state_change_fn, model_cls=BrainScoreModel):
    return model_cls(
        identifier="native-state-change",
        model=None,
        region_layer_map={},
        preprocessors={"vision": lambda stimuli: stimuli},
        state_change_fn=state_change_fn,
    )


def _native_environment_model(action_fn, model_cls=BrainScoreModel):
    return model_cls(
        identifier="native-environment",
        model=None,
        region_layer_map={},
        preprocessors={"vision": lambda stimuli: stimuli},
        policy=action_fn,
        policy_inputs=("vision:image", "proprioception", "instruction"),
    )


def _native_multimodal_model(model_cls=BrainScoreModel):
    activations = _XArrayActivationsModel()
    text = _XArrayPreprocessor(label="text")
    model = model_cls(
        identifier="native-multimodal",
        model=None,
        region_layer_map={"IT": "layer.10"},
        preprocessors={
            "vision": make_stub_preprocessor(),
            "text": text,
        },
        activations_model=activations,
    )
    return model, activations, text


def _native_cross_tower_model(model_cls=BrainScoreModel):
    activations = _XArrayActivationsModel()
    text = _XArrayPreprocessor(label="text")
    model = model_cls(
        identifier="native-cross-tower",
        model=None,
        region_layer_map={
            "IT": "vision_model.layer.10",
            "language_network": "text_model.layer.20",
        },
        preprocessors={
            "vision": make_stub_preprocessor(),
            "text": text,
        },
        activations_model=activations,
        region_modality_map={
            "IT": "vision",
            "language_network": "text",
        },
    )
    return model, activations, text


class _TemporalBrainScoreModel(BrainScoreModel):
    def __init__(self):
        super().__init__(
            identifier="native-temporal",
            model=None,
            region_layer_map={"IT": "temporal_layer"},
            preprocessors={"vision": lambda stimuli: stimuli},
        )

    def process(self, input_event, multi_modality=False):
        del multi_modality
        stimulus_ids = list(input_event["stimulus_id"].values)
        neuroid_ids = ["temporal_layer.0", "temporal_layer.1"]
        if self._time_bins is None:
            return NeuroidAssembly(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                coords={
                    "stimulus_id": ("presentation", stimulus_ids),
                    "neuroid_id": ("neuroid", neuroid_ids),
                    "layer": ("neuroid", ["temporal_layer"] * 2),
                },
                dims=["presentation", "neuroid"],
            )

        data = np.arange(
            len(stimulus_ids) * len(self._time_bins) * 2, dtype=float
        ).reshape(len(stimulus_ids), len(self._time_bins), 2)
        return NeuroidAssembly(
            data,
            coords={
                "stimulus_id": ("presentation", stimulus_ids),
                "time_bin_start_ms": (
                    "time_bin", [start for start, _ in self._time_bins]
                ),
                "time_bin_end_ms": (
                    "time_bin", [end for _, end in self._time_bins]
                ),
                "neuroid_id": ("neuroid", neuroid_ids),
                "layer": ("neuroid", ["temporal_layer"] * 2),
            },
            dims=["presentation", "time_bin", "neuroid"],
        )


class _TrackingTemporalBrainScoreModel(_TemporalBrainScoreModel):
    def __init__(self):
        super().__init__()
        self.participate_called = False
        self.participate_requested_output_channels = None

    def participate(self, session):
        self.participate_called = True
        self.participate_requested_output_channels = tuple(
            session.requested_output_channels
        )
        return super().participate(session)


def test_neural_response_uses_native_participate_and_preserves_metadata_exactly():
    stimuli = _metadata_stimulus_set()

    legacy_extractor = _MetadataPreservingExtractor()
    legacy = _native_neural_model(legacy_extractor)
    legacy.start_recording("IT")
    expected = legacy.process(stimuli)

    native_extractor = _MetadataPreservingExtractor()
    subject = _native_neural_model(
        native_extractor, model_cls=_ParticipateTrackingBrainScoreModel
    )
    scored = neural_response(subject, stimuli, record="IT")

    assert subject.participate_called is True
    assert subject.participate_requested_output_channels == ("neural:IT",)
    assert native_extractor.last_stimuli is stimuli
    assert "modality" not in scored.coords
    xr.testing.assert_identical(scored, expected)


def test_neural_response_native_participate_matches_multimodal_process_exactly():
    stimuli = _multimodal_stimulus_set()

    legacy, _, _ = _native_multimodal_model()
    legacy.start_recording("IT")
    expected = legacy.process(stimuli, multi_modality=True)

    subject, activations, text = _native_multimodal_model(
        model_cls=_ParticipateTrackingBrainScoreModel
    )
    scored = neural_response(subject, stimuli, record="IT")

    assert subject.participate_called is True
    assert subject.participate_requested_output_channels == ("neural:IT",)
    assert activations.call_args is not None
    assert text.call_args is not None
    assert set(scored["modality"].values.tolist()) == {"vision", "text"}
    xr.testing.assert_identical(scored, expected)


def test_neural_response_native_participate_matches_cross_tower_process_exactly():
    stimuli = _multimodal_stimulus_set()

    legacy, _, _ = _native_cross_tower_model()
    legacy.start_recording("all")
    expected = legacy.process(stimuli, multi_modality=True)

    subject, activations, text = _native_cross_tower_model(
        model_cls=_ParticipateTrackingBrainScoreModel
    )
    scored = neural_response(subject, stimuli, record="all")

    assert subject.participate_called is True
    assert subject.participate_requested_output_channels == ("neural:all",)
    assert activations.call_args["layers"] == ["vision_model.layer.10"]
    assert text.call_args["layers"] == ["text_model.layer.20"]
    assert set(scored["modality"].values.tolist()) == {"vision", "text"}
    xr.testing.assert_identical(scored, expected)


def test_neural_response_native_participate_threads_time_bins_exactly():
    stimuli = _metadata_stimulus_set()
    time_bins = [(70, 170), (170, 270)]

    legacy = _TemporalBrainScoreModel()
    legacy.start_recording("IT", time_bins=time_bins)
    expected = legacy.process(stimuli)

    subject = _TrackingTemporalBrainScoreModel()
    scored = neural_response(subject, stimuli, record="IT", time_bins=time_bins)

    assert subject.participate_called is True
    assert subject.participate_requested_output_channels == ("neural:IT",)
    assert scored.dims == ("presentation", "time_bin", "neuroid")
    assert list(scored["time_bin_start_ms"].values) == [70, 170]
    assert list(scored["time_bin_end_ms"].values) == [170, 270]
    xr.testing.assert_identical(scored, expected)


def test_neural_response_without_time_bins_keeps_2d_output():
    stimuli = _metadata_stimulus_set()

    legacy = _TemporalBrainScoreModel()
    legacy.start_recording("IT")
    expected = legacy.process(stimuli)

    subject = _TrackingTemporalBrainScoreModel()
    scored = neural_response(subject, stimuli, record="IT")

    assert subject.participate_called is True
    assert subject.participate_requested_output_channels == ("neural:IT",)
    assert scored.dims == ("presentation", "neuroid")
    assert "time_bin" not in scored.dims
    xr.testing.assert_identical(scored, expected)


def test_event_only_reconstruction_is_input_columns_only():
    events = [
        StreamEvent(
            channel="vision",
            payload="s0.png",
            t_ms=0.0,
            meta={
                "stimulus_id": "s0",
                "stimulus_index": 0,
                "column": "image_path",
                "object_name": "cat",
            },
        ),
        StreamEvent(
            channel="text",
            payload="cat sentence",
            t_ms=0.0,
            meta={
                "stimulus_id": "s0",
                "stimulus_index": 0,
                "column": "sentence",
                "object_name": "cat",
            },
        ),
    ]

    reconstructed = BrainScoreModel._reconstruct_stimulus_set_from_events(events)

    assert list(reconstructed["stimulus_id"].values) == ["s0"]
    assert list(reconstructed["image_path"].values) == ["s0.png"]
    assert list(reconstructed["sentence"].values) == ["cat sentence"]
    assert "object_name" not in reconstructed.columns


def _behavior_model(features, model_cls=BrainScoreModel):
    return model_cls(
        identifier="test-model",
        model=None,
        region_layer_map={},
        preprocessors={"vision": _fake_preprocessor_returning(features)},
        behavioral_readout_layer="some_layer",
    )


def _behavior_context(fitting_stimuli, scoring_stimuli):
    return TaskContext(
        task_type="probabilities",
        fitting_stimuli=fitting_stimuli,
        label_set=["cat", "dog"],
        instruction="Choose the animal",
        metadata={"stimulus_set": scoring_stimuli},
    )


def test_behavioral_response_matches_legacy_start_task_process_exactly():
    fitting = _make_image_stimulus_set(["cat"] * 6 + ["dog"] * 6)
    scoring = _make_image_stimulus_set(
        ["cat"] * 3 + ["dog"] * 3, identifier="behavior_score"
    )
    features = np.random.default_rng(42).normal(size=(12, 8))
    features[:6, 0] += 3.0
    features[6:, 0] -= 3.0

    legacy = _behavior_model(features)
    legacy_context = _behavior_context(fitting, scoring)
    legacy.start_task(legacy_context)
    expected = legacy.process(scoring)

    helper_subject = _behavior_model(
        features, model_cls=_ParticipateTrackingBrainScoreModel
    )
    helper_context = _behavior_context(fitting, scoring)
    scored = behavioral_response(helper_subject, helper_context)

    assert helper_subject.participate_called is True
    assert helper_subject.participate_requested_output_channels == ("behavior",)
    assert isinstance(scored, BehavioralAssembly)
    xr.testing.assert_identical(scored, expected)


def _generation_model(calls, model_cls=BrainScoreModel):
    def fake_generate(stimulus_row, instruction, label_set):
        calls.append((stimulus_row["stimulus_id"], instruction, tuple(label_set)))
        return "cat" if stimulus_row["stimulus_id"] in {"s0", "s1", "s2"} else "dog"

    return model_cls(
        identifier="test-vlm",
        model=None,
        region_layer_map={},
        preprocessors={"vision": lambda *args, **kwargs: None},
        generation_fn=fake_generate,
    )


def _generation_context(scoring_stimuli):
    return TaskContext(
        task_type="probabilities",
        label_set=["cat", "dog"],
        instruction="What animal is this?",
        metadata={"stimulus_set": scoring_stimuli},
    )


def test_behavioral_response_matches_legacy_generation_path_exactly():
    scoring = _make_image_stimulus_set(
        ["cat"] * 3 + ["dog"] * 3, identifier="gen_test"
    )

    legacy_calls = []
    legacy = _generation_model(legacy_calls)
    legacy_context = _generation_context(scoring)
    legacy.start_task(legacy_context)
    expected = legacy.process(scoring)

    helper_calls = []
    helper_subject = _generation_model(
        helper_calls, model_cls=_ParticipateTrackingBrainScoreModel
    )
    helper_context = _generation_context(scoring)
    scored = behavioral_response(helper_subject, helper_context)

    assert helper_subject.participate_called is True
    assert helper_subject.participate_requested_output_channels == ("behavior",)
    assert isinstance(scored, BehavioralAssembly)
    xr.testing.assert_identical(scored, expected)
    assert helper_calls == legacy_calls


class _SyntheticEnvironment:
    """``count`` rows: reset gives the first; the last is terminal."""

    def __init__(self, count):
        self.count = count
        self.actions = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return _droid_row("reach the goal")

    def step(self, outputs):
        self.actions.append(outputs['motor'])
        last = len(self.actions) == self.count - 1
        return _droid_row("reach the goal"), 'terminal' if last else None


def _counting_policy():
    """A zero-action policy that echoes how many rows it has seen."""
    calls = []

    def policy(row):
        calls.append(row)
        return {'motor': StreamEvent('motor', np.zeros(7, dtype=np.float64), None, {
            'step_num_seen': len(calls) - 1, 'instruction_seen': row['instruction']})}
    return policy


def _legacy_environment_rollout(subject, environment):
    """The direct loop: process each row, apply the action, stop at the terminal row."""
    trajectory = []
    row = environment.reset()
    while True:
        outputs = subject.process(row)
        trajectory.append(outputs['motor'])
        row, reason = environment.step({'motor': outputs['motor'].payload})
        if reason is not None:
            return trajectory


def test_native_environment_participate_preserves_applied_legacy_actions():
    session = environment_session(_SyntheticEnvironment(4))
    native_model = _native_environment_model(
        _counting_policy(), model_cls=_ParticipateTrackingBrainScoreModel
    )

    native_model.participate(session)
    native_trajectory = session.collect("motor")

    legacy_model = _make_environment_model(policy=_counting_policy())
    expected_trajectory = _legacy_environment_rollout(legacy_model, _SyntheticEnvironment(4))

    assert native_model.participate_called is True
    assert native_model.participate_requested_output_channels == ("motor",)
    assert [event.channel for event in session.emitted] == [
        "motor", "motor", "motor",
    ]
    assert [event.meta["driver"] for event in session.emitted] == [
        "interact", "interact", "interact",
    ]
    assert [event.meta["step_num"] for event in session.emitted] == [0, 1, 2]
    assert len(native_trajectory) == len(expected_trajectory)
    for native_response, expected_response in zip(
        native_trajectory, expected_trajectory
    ):
        np.testing.assert_array_equal(
            native_response.payload, expected_response.payload
        )
        assert native_response.meta["step_num_seen"] == expected_response.meta["step_num_seen"]


def test_run_environment_preserves_applied_legacy_actions():
    helper_model = _native_environment_model(
        _counting_policy(), model_cls=_ParticipateTrackingBrainScoreModel
    )
    helper_env = _SyntheticEnvironment(4)
    helper_trajectory = run_environment(helper_model, helper_env)

    legacy_model = _make_environment_model(policy=_counting_policy())
    legacy_env = _SyntheticEnvironment(4)
    expected_trajectory = _legacy_environment_rollout(legacy_model, legacy_env)

    assert len(helper_trajectory) == len(expected_trajectory)
    for helper_action, expected_response in zip(helper_trajectory, expected_trajectory):
        np.testing.assert_array_equal(helper_action, expected_response.payload)
    assert len(helper_env.actions) == len(legacy_env.actions)
    for helper_action, expected_action in zip(helper_env.actions, legacy_env.actions):
        np.testing.assert_array_equal(helper_action, expected_action)
    assert helper_model.participate_called is True
    assert helper_model.participate_requested_output_channels == ("motor",)


def test_run_environment_terminates_cleanly_for_a_one_step_env():
    model = _native_environment_model(
        _counting_policy(), model_cls=_ParticipateTrackingBrainScoreModel
    )
    one_env = _SyntheticEnvironment(2)
    trajectory = run_environment(model, one_env)

    assert len(trajectory) == 1 and len(one_env.actions) == 1  # the terminal row asks for no action
    assert model.participate_called is True
    assert model.participate_requested_output_channels == ("motor",)


def _ablation_state_change():
    return StateChange(
        kind="ablation",
        target=Selection(layer="blocks.10", indices=[1, 2, 3]),
        perturbation=Perturbation(kind="zero"),
    )


def test_apply_state_change_matches_legacy_process_result():
    fn_legacy, _ = _counting_state_change_fn()
    legacy = _make_state_change_model(state_change_fn=fn_legacy)
    legacy_result = legacy.process(_ablation_state_change())

    fn_helper, _ = _counting_state_change_fn()
    helper_model = _native_state_change_model(
        fn_helper, model_cls=_ParticipateTrackingBrainScoreModel
    )
    helper_result = apply_state_change(helper_model, _ablation_state_change())

    assert helper_model.participate_called is True
    assert helper_model.participate_requested_output_channels == ("perturbation",)
    assert isinstance(helper_result, PerturbationApplied)
    assert helper_result.handle_id == legacy_result.handle_id
    assert helper_result.target == legacy_result.target
    assert helper_result.perturbation == legacy_result.perturbation
    assert helper_result.applied_at == legacy_result.applied_at


def test_native_state_change_reset_restores_baseline():
    model_state = {"output": 1.0}

    def state_change_fn(state_change):
        saved = model_state["output"]
        model_state["output"] = 0.0
        applied = PerturbationApplied(
            handle_id="ablation-handle",
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup():
            model_state["output"] = saved

        return applied, cleanup

    model = _native_state_change_model(
        state_change_fn, model_cls=_ParticipateTrackingBrainScoreModel
    )
    applied = apply_state_change(model, _ablation_state_change())
    assert model_state["output"] == 0.0
    assert model.participate_requested_output_channels == ("perturbation",)

    reset = StateChange(kind="reset", handle_id=applied.handle_id)
    reset_session = state_change_session(reset)
    reset_event = reset_session.next_input()
    model.participate_called = False
    reset_result = apply_state_change(model, reset)

    assert model.participate_called is True
    assert model.participate_requested_output_channels == ("perturbation",)
    assert reset_event.channel == "lesion:ablation-handle"
    assert reset_event.meta["reset"] == applied.handle_id
    assert reset_result == applied.handle_id
    assert model_state["output"] == 1.0


# --- moved from core (need BrainScoreModel) -------------------------------


def test_environment_session_emits_motor_events_for_multistep_rollout():
    session = environment_session(_SyntheticEnvironment(3))
    model = _make_environment_model(policy=_counting_policy())

    from brainscore_core.streaming_helpers import drive_environment_session
    drive_environment_session(model, session)

    assert [event.channel for event in session.emitted] == [
        "motor", "motor",
    ]
    assert [event.meta["step_num_seen"] for event in session.emitted] == [0, 1]
    assert [event.meta["step_num"] for event in session.emitted] == [0, 1]


def test_apply_state_change_produces_handle_and_ack_event():
    fn, state = _counting_state_change_fn()
    model = _make_state_change_model(state_change_fn=fn)
    state_change = _ablation_state_change()

    session = state_change_session(state_change)
    _ = session.next_input()
    from brainscore_core.streaming_helpers import _drive_state_change_via_process
    _drive_state_change_via_process(model, session, state_change)

    applied = session.collect("perturbation")

    assert isinstance(applied, PerturbationApplied)
    assert applied.handle_id in state["installed"]
    assert session.emitted[0].meta["handle_id"] == applied.handle_id
    assert session.emitted[0].payload is applied
