import numpy as np
import pytest

pytestmark = pytest.mark.unit


class _Frames:
    """A made-up game: 5 steps of random frames, the shape the mario environment gives."""

    def __init__(self):
        self.rng, self.steps, self.actions = np.random.default_rng(0), 0, []

    def _row(self):
        return {'vision:video': self.rng.integers(0, 255, (4, 224, 240, 3), dtype=np.uint8),
                'audio': np.zeros(2136, np.float32), 'instruction': 'Play.'}

    def reset(self, **_):
        self.steps = 0
        return self._row()

    def step(self, outputs):
        self.actions.append(outputs['motor'])
        self.steps += 1
        return self._row(), ('terminal' if self.steps == 5 else None)


def test_the_agent_plays_and_its_layers_are_recorded_on_the_input_it_acted_on():
    from brainscore import load_model
    from brainscore_core.streaming_helpers import environment_session
    model, game = load_model('mario-ppo-untrained'), _Frames()
    session = environment_session(game, ['motor', 'neural:cortex'])
    model.participate(session)
    assert len(game.actions) == 5 and all(0 <= action < 12 for action in game.actions)
    recorded = session.collect('neural:cortex')
    assert recorded.sizes['presentation'] == 5          # one per action: the last row ends the play
    layers = np.asarray(recorded['layer'].values).astype(str)
    assert sorted(set(layers)) == ['conv2', 'conv3', 'conv4', 'fc'] and (layers == 'fc').sum() == 512


def test_the_untrained_weights_are_the_same_each_time_and_the_trained_source_is_pinned():
    from brainscore.models.mario_ppo import model
    first, second = model.build_network(), model.build_network()
    assert all(np.array_equal(a.detach().numpy(), b.detach().numpy())
               for a, b in zip(first.parameters(), second.parameters()))
    assert model.COMMIT in model.URL and len(model.SHA256) == 64


def test_the_input_is_the_last_four_grey_frames_at_84_by_84():
    from brainscore.models.mario_ppo.model import grey
    frame = np.zeros((224, 240, 3), np.uint8)
    frame[:, :120] = 255
    image = grey(frame)
    assert image.shape == (84, 84) and image[0, 0] == 255 and image[0, -1] == 0


def test_one_step_runs_the_network_once_and_the_policy_declares_only_what_it_reads(monkeypatch):
    from brainscore.models.mario_ppo import model as mario
    from brainscore_core.streaming_helpers import environment_session
    calls, build = [], mario.build_network

    def counted():
        network = build()
        network.conv1.register_forward_hook(lambda *_: calls.append(1))
        return network
    monkeypatch.setattr(mario, 'build_network', counted)
    subject, game = mario.get_model('mario-ppo-untrained'), _Frames()
    subject.participate(environment_session(game, ['motor', 'neural:cortex']))
    assert len(calls) == len(game.actions) == 5
    assert 'audio' not in subject.in_channels and 'instruction' not in subject.in_channels
