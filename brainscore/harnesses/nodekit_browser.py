"""Closed-loop harness that runs a built nodekit site in headless Chromium.

Humans and models play the *same* page: nodekit's browser runtime logs pointer
samples (``PointerSampledEvent``, Board coordinates, moves capped at 30 Hz) for
both, so model and human traces come out of one logger in one format.

The model acts through the ordinary embodied path, ``process(EnvironmentStep)``
-> ``EnvironmentResponse``. Its ``action`` is a pointer action: an ``(n, 4)``
array of rows ``(dt_ms, x, y, kind)``. ``dt_ms`` is page-clock time to let pass
before the sample, ``(x, y)`` is in Board coordinates (origin at the centre,
y up) and ``kind`` is 0 = move, 1 = button down, 2 = button up. Build one with
:func:`pointer_action`.

Timing: the page clock is paused (Playwright's clock API) between actions, so
the time a model spends thinking is not recorded as reaction time. Page time
only advances by the ``dt_ms`` the model emits (plus ``settle_ms`` after each
action so the page can render the next node). A model that emits a single
click with ``dt_ms=0`` jumps to the point instantly; that is recorded as is.

Requires ``playwright`` and its Chromium (``pip install playwright &&
playwright install chromium``). Sites are built with nodekit in a separate
Python 3.12 environment; see ``brainscore/benchmarks/nodekit_fitts``.
"""
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from brainscore_core.model_interface import EnvironmentStep

MOVE, DOWN, UP = 0, 1, 2
BOARD_SIZE = 1024  # nodekit's normative board, in CSS px

# Wrap window.NodeKit.play so every trace event is also pushed to window.__nk.
# The built site calls NodeKit.play(graph) without an event callback.
_CAPTURE_SCRIPT = """
window.__nk = [];
let _nk;
Object.defineProperty(window, 'NodeKit', {configurable: true,
  get() { return _nk; },
  set(v) {
    const play = v.play;
    _nk = Object.assign({}, v, {play: (g, cb, d) => play(g, e => {
      window.__nk.push(e); if (cb) cb(e); }, d)});
  }});
"""

_CARDS_SCRIPT = """() => {
  const board = document.querySelector('.board-view');
  if (!board) return null;
  const b = board.getBoundingClientRect();
  return {board: [b.left, b.top, b.width, b.height],
          cards: [...board.querySelectorAll('.board-region')].map(e => {
            const r = e.getBoundingClientRect();
            return [r.left - b.left + r.width / 2 - b.width / 2,
                    b.height / 2 - (r.top - b.top + r.height / 2), r.width, r.height];
          })};
}"""


def pointer_action(samples: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate rows ``(dt_ms, x, y, kind)`` into the ``(n, 4)`` pointer action."""
    action = np.asarray(samples, dtype=float).reshape(-1, 4)
    if np.any(action[:, 0] < 0):
        raise ValueError('dt_ms must be >= 0')
    if not np.isin(action[:, 3], (MOVE, DOWN, UP)).all():
        raise ValueError('kind must be 0 (move), 1 (down) or 2 (up)')
    return action


def click(x: float, y: float, dt_ms: float = 0.0, hold_ms: float = 0.0) -> np.ndarray:
    """A click at Board (x, y): move there, press, release."""
    return pointer_action([(dt_ms, x, y, MOVE), (0, x, y, DOWN), (hold_ms, x, y, UP)])


class NodekitBrowserEnvironment:
    """``reset()`` / ``step(action)`` over one built nodekit site.

    Observations are dicts: ``image`` (the board screenshot, uint8 RGB,
    ``BOARD_SIZE`` square), ``cursor`` (last Board position), ``t_ms`` (page
    clock), ``node_address`` and ``instruction``. ``_cards`` lists the visible
    cards as ``(x, y, w, h)`` Board rectangles; it is privileged (reference
    policies use it, models should not).

    :param site_html: path to the built site's entry HTML.
    :param max_steps: actions allowed per episode before it is cut off.
    :param settle_ms: page time run after each action so the next node renders.
    :param nodekit_version: recorded in :meth:`trace` (the runtime does not expose it).
    :param move_gap_ms: real wall-clock gap between dispatched moves. nodekit
        throttles moves to 30 Hz on the event's real timestamp, which the paused
        page clock does not control, so moves sent faster are dropped.
    """

    def __init__(self, site_html: Union[str, Path], *, instruction: str = '',
                 max_steps: int = 200, settle_ms: float = 50.0,
                 move_gap_ms: float = 35.0, headless: bool = True,
                 screenshots: bool = True, nodekit_version: Optional[str] = None):
        self.site_html = Path(site_html).resolve()
        if not self.site_html.exists():
            raise FileNotFoundError(self.site_html)
        self.instruction = instruction
        self.max_steps = max_steps
        self.settle_ms = settle_ms
        self.move_gap_ms = move_gap_ms
        self.headless = headless
        self.screenshots = screenshots
        self.nodekit_version = nodekit_version
        self._pw = self._browser = self._page = None
        self._steps = 0
        self._cursor = (0.0, 0.0)
        self._board = None

    # -- lifecycle -------------------------------------------------------
    def _ensure_browser(self):
        if self._browser is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as e:
                raise RuntimeError('pip install playwright && playwright install chromium') from e
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless)

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._pw.stop()
        self._pw = self._browser = self._page = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self) -> EnvironmentStep:
        self._ensure_browser()
        if self._page is not None:
            self._page.close()
        side = BOARD_SIZE + 76  # the board plus nodekit's margins, so 1 unit = 1 px
        page = self._browser.new_page(viewport={'width': side + 256, 'height': side})
        page.add_init_script(_CAPTURE_SCRIPT)
        # Freeze page time before the page loads, so every page time is
        # reproducible. Python's pause_at takes seconds since the epoch.
        page.clock.install(time=0)
        page.clock.pause_at(1)
        page.goto(self.site_html.as_uri())
        page.locator('#session-started-overlay button').click()
        self._page = page
        self._steps = 0
        self._cursor = (0.0, 0.0)
        self._run(self.settle_ms)
        return self._observe(is_first=True)

    def step(self, action) -> EnvironmentStep:
        rows = pointer_action(action)
        page = self._page
        for dt, x, y, kind in rows:
            self._run(dt)
            px, py = self._to_page(x, y)
            if kind == MOVE:
                page.mouse.move(px, py)
                page.wait_for_timeout(self.move_gap_ms)  # real time; see move_gap_ms
            else:
                if (x, y) != self._cursor:
                    page.mouse.move(px, py)
                (page.mouse.down if kind == DOWN else page.mouse.up)()
            self._cursor = (float(x), float(y))
        self._steps += 1
        self._run(self.settle_ms)
        return self._observe()

    # -- reading the page ------------------------------------------------
    def events(self) -> List[Dict[str, Any]]:
        return self._page.evaluate('window.__nk')

    def trace(self) -> Dict[str, Any]:
        """The nodekit trace so far, in nodekit's ``Trace`` JSON shape (no graph)."""
        return {'nodekit_version': self.nodekit_version, 'events': self.events()}

    def _run(self, ms: float) -> None:
        if ms > 0:
            self._page.clock.run_for(int(round(ms)))

    def _to_page(self, x: float, y: float) -> Tuple[float, float]:
        left, top, width, height = self._board
        return left + width / 2 + x, top + height / 2 - y

    def _observe(self, is_first: bool = False) -> EnvironmentStep:
        page = self._page
        layout = page.evaluate(_CARDS_SCRIPT)
        events = self.events()
        ended = any(e['event_type'] == 'TraceEndedEvent' for e in events)
        if layout is not None:
            self._board = layout['board']
        nodes = [e for e in events if e['event_type'] == 'NodeStartedEvent']
        t_ms = float(page.evaluate('performance.now()'))
        image = None
        if self.screenshots and layout is not None and not ended:
            from PIL import Image
            png = page.locator('.board-view').screenshot()
            image = np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
        observation = {
            'image': image,
            'cursor': self._cursor,
            't_ms': t_ms,
            'node_address': tuple(nodes[-1]['node_address']) if nodes else (),
            'instruction': self.instruction,
            '_cards': [tuple(c) for c in (layout or {}).get('cards', [])],
        }
        last = ended or self._steps >= self.max_steps
        return EnvironmentStep(
            observation=observation, instruction=self.instruction,
            is_first=is_first, is_last=last, is_terminal=ended,
            step_num=self._steps,
            context={'t_ms': t_ms, 'time_source': 'nodekit_page_clock',
                     'time_inferred': False})


# -- reference and null policies: policy(observation, history) -> action ---

def random_pointer_policy(seed: int = 0, n_moves: int = 6, dt_ms: float = 40.0,
                          extent: float = BOARD_SIZE / 2) -> Callable:
    """The null: wander through ``n_moves`` random points, then click the last.

    It never looks at the screen, so any score it earns is the floor.
    """
    rng = np.random.RandomState(seed)

    def policy(observation, history):
        pts = rng.uniform(-extent, extent, size=(n_moves, 2))
        rows = [(dt_ms, x, y, MOVE) for x, y in pts]
        x, y = pts[-1]
        return pointer_action(rows + [(0, x, y, DOWN), (dt_ms, x, y, UP)])

    return policy


def straight_reach_policy(speed: float = 1.0, dt_ms: float = 40.0,
                          card: int = -1) -> Callable:
    """Reference mover: straight line to a card's centre at ``speed`` px/ms, then click.

    Reads the privileged ``_cards`` layout. It validates the harness and
    metrics end to end; it is not a model of human reaching.
    """
    def policy(observation, history):
        cards = observation['_cards']
        if not cards:
            return pointer_action([(dt_ms, 0, 0, MOVE)])
        tx, ty = cards[card][:2]
        sx, sy = observation['cursor']
        n = max(1, int(np.ceil(np.hypot(tx - sx, ty - sy) / (speed * dt_ms))))
        rows = [(dt_ms, sx + (tx - sx) * k / n, sy + (ty - sy) * k / n, MOVE)
                for k in range(1, n + 1)]
        return pointer_action(rows + [(0, tx, ty, DOWN), (dt_ms, tx, ty, UP)])

    return policy


def play_site(model, environment: NodekitBrowserEnvironment) -> Dict[str, Any]:
    """Run one episode through ``model.process(EnvironmentStep)``; return the trace.

    Uses core's ``run_environment`` so the pointer actions travel on the
    ``motor`` channel like any other embodied action.
    """
    from brainscore_core.streaming_helpers import run_environment
    run_environment(model, environment)
    return environment.trace()
