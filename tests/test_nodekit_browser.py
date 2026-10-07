"""Integration tests: pointer actions replayed in headless Chromium on the
committed nodekit Fitts site. Offline, but needs ``playwright`` and its
Chromium (``playwright install chromium``); skipped when they are missing."""
import time

import numpy as np
import pytest

pytest.importorskip('playwright')

from brainscore.benchmarks.nodekit_fitts.benchmark import load_design  # noqa: E402
from brainscore.cursor_traces import read_trials  # noqa: E402
from brainscore.harnesses.nodekit_browser import (  # noqa: E402
    BOARD_SIZE, MOVE, NodekitBrowserEnvironment, click, play_site, pointer_action, straight_reach_policy)
from brainscore.model_helpers.pointer_policy import build_click_policy  # noqa: E402
from brainscore.models.nodekit_pointer import pointer_model  # noqa: E402

pytestmark = pytest.mark.integration

DESIGN, SITE = load_design()


@pytest.fixture(scope='module', autouse=True)
def chromium():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            p.chromium.launch().close()
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                pytest.skip('Chromium not installed: playwright install chromium')
            raise


def _play(policy, max_steps=2):
    with NodekitBrowserEnvironment(SITE, max_steps=max_steps) as env:
        return play_site(pointer_model('test', policy), env)


def test_samples_are_logged_where_and_when_they_were_sent():
    with NodekitBrowserEnvironment(SITE) as env:
        env.reset()
        env.step(click(0, 0, dt_ms=20))                      # home button
        target = DESIGN['trials'][0]
        path = [(40, target['x'] * k / 4, target['y'] * k / 4, MOVE) for k in range(1, 5)]
        env.step(pointer_action(path))
        trials = read_trials(env.trace())
    home, first = trials[0], trials[1]
    assert home.action['action_value'] == 'home' and first.node_id == 'target'
    moves = first.samples[first.samples[:, 3] == MOVE]
    np.testing.assert_allclose(moves[:, 1:3], [p[1:3] for p in path], atol=1)
    np.testing.assert_allclose(np.diff(moves[:, 0]), 40)     # page clock follows dt_ms


def test_model_thinking_time_is_not_recorded():
    def slow(policy):
        def wrapped(observation, history):
            time.sleep(0.3)
            return policy(observation, history)
        return wrapped
    fast = _play(straight_reach_policy())
    slowed = _play(slow(straight_reach_policy()))
    assert [e['t'] for e in fast['events']] == [e['t'] for e in slowed['events']]


def test_click_policy_completes_a_trial_from_pixels():
    target = DESIGN['trials'][0]
    half = BOARD_SIZE / 2

    def generate(image, prompt):
        assert image.shape == (BOARD_SIZE, BOARD_SIZE, 3)
        # Answer in screenshot pixels: the home button, then the target.
        x, y = (0, 0) if generate.calls == 0 else (target['x'], target['y'])
        generate.calls += 1
        return f'Click: ({half + x}, {half - y})'
    generate.calls = 0
    trials = read_trials(_play(build_click_policy(generate)))
    assert [t.action['action_value'] for t in trials if t.completed] == ['home', 'target']


def test_target_is_dark_on_the_screenshot():
    target = DESIGN['trials'][0]
    with NodekitBrowserEnvironment(SITE) as env:
        env.reset()
        image = env.step(click(0, 0, dt_ms=20)).observation['image']
    half = BOARD_SIZE // 2
    assert image[half - target['y'], half + target['x']].max() < 100   # target centre
    assert image[10, 10].min() > 240                                    # white board
