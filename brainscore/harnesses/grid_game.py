"""Grid navigation environment and policy examples.

GridGameEnvironment declares observations/actions for an EnvironmentSource.
The oracle reads privileged _state; visual policies use the rendered frame.
"""
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from brainscore_core.environment import ArraySpec, DiscreteSpec, MappingSpec, TextSpec
from brainscore_core.streaming_helpers import run_environment

# Discrete action space. Row 0 is the top of the grid, so "up" decreases row.
ACTIONS: Dict[int, str] = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
_DELTA: Dict[int, Tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# Render palette (R, G, B).
_BG = (235, 235, 235)
_GRID = (200, 200, 200)
_AGENT = (40, 90, 220)     # blue player
_GOAL = (40, 190, 70)      # green goal
_WALL = (30, 30, 30)


class GridGameEnv:
    """A tiny grid-navigation video game.

    :param size: grid side length (``size`` x ``size`` cells).
    :param n_walls: number of randomly placed impassable wall cells.
    :param seed: RNG seed for agent/goal/wall placement (reproducible).
    :param max_steps: episode length cap before forced termination.
    :param cell_px: pixels per cell in the rendered frame.
    """

    def __init__(self, size: int = 6, n_walls: int = 0, seed: int = 0,
                 max_steps: int = 40, cell_px: int = 32):
        self.size = size
        self.n_walls = n_walls
        self.max_steps = max_steps
        self.cell_px = cell_px
        self._rng = np.random.RandomState(seed)
        self.instruction = ("Navigate the blue player to the green goal. "
                            "Reply with one word: up, down, left, or right.")
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (size - 1, size - 1)
        self.walls: set = set()
        self._steps = 0
        self.reset()

    # -- environment API -------------------------------------------------
    def reset(self, *, seed=None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self._rng.shuffle(cells)
        self.agent_pos = cells[0]
        self.goal_pos = cells[1]
        self.walls = set(cells[2:2 + self.n_walls])
        self._steps = 0
        return self._observe()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        action = DiscreteSpec(len(ACTIONS)).validate(action)
        self._steps += 1
        dr, dc = _DELTA[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        moved = False
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            self.agent_pos = (nr, nc)
            moved = True
        reached = self.agent_pos == self.goal_pos
        if reached:
            reward, done = 1.0, True
        else:
            reward = -0.01
            done = self._steps >= self.max_steps
        info = {'moved': moved, 'reached_goal': reached, 'steps': self._steps,
                'distance': self._manhattan()}
        return self._observe(), reward, done, info

    def render(self) -> np.ndarray:
        """Render the board to an ``(H, W, 3) uint8`` RGB array."""
        k, n = self.cell_px, self.size
        img = np.zeros((n * k, n * k, 3), dtype=np.uint8)
        img[:, :] = _BG
        # grid lines
        for i in range(n + 1):
            img[min(i * k, n * k - 1), :] = _GRID
            img[:, min(i * k, n * k - 1)] = _GRID

        def fill(cell, color, pad=4):
            r, c = cell
            img[r * k + pad:(r + 1) * k - pad, c * k + pad:(c + 1) * k - pad] = color

        for w in self.walls:
            fill(w, _WALL, pad=1)
        fill(self.goal_pos, _GOAL)
        fill(self.agent_pos, _AGENT)
        return img

    def ascii_board(self) -> str:
        """Text rendering of the board: ``P`` player, ``G`` goal, ``#`` wall,
        ``.`` empty. Gives a *thinking* text model perfect perception so the
        closed loop tests reasoning, not vision — the complement to ``render``.
        """
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    cells.append('P')
                elif (r, c) == self.goal_pos:
                    cells.append('G')
                elif (r, c) in self.walls:
                    cells.append('#')
                else:
                    cells.append('.')
            rows.append(' '.join(cells))
        return '\n'.join(rows)

    # -- helpers ---------------------------------------------------------
    def _manhattan(self) -> int:
        return (abs(self.agent_pos[0] - self.goal_pos[0])
                + abs(self.agent_pos[1] - self.goal_pos[1]))

    def _observe(self) -> Dict[str, Any]:
        return {
            'frame': self.render(),
            'ascii': self.ascii_board(),
            'instruction': self.instruction,
            'legal_actions': dict(ACTIONS),
            # Privileged ground truth — oracle only; visual policies must ignore
            # keys beginning with '_'.
            '_state': {'agent': self.agent_pos, 'goal': self.goal_pos,
                       'size': self.size, 'walls': list(self.walls)},
        }


class GridGameEnvironment:
    """Wrap grid observations and actions for one turn-based environment session.

    The rendered view and privileged oracle state are unchanged. There is no
    physical clock: steps count decisions and t_ms is unknown.
    """

    def __init__(self, env: GridGameEnv, *, max_steps=None):
        self.env = env
        self.max_steps = env.max_steps if max_steps is None else max_steps
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self._step_num = 0
        self.total_reward = 0.
        self.solved = False
        self.optimal_steps = None

    def observation_spec(self):
        pixels = self.env.size * self.env.cell_px
        return MappingSpec({
            'frame': ArraySpec((pixels, pixels, 3), 'uint8'),
            'instruction': TextSpec(),
            'ascii': TextSpec(),
            'legal_actions': MappingSpec({key: TextSpec() for key in ACTIONS}),
        }, allow_extra=True)  # _state is privileged input for the oracle demo.

    def action_spec(self):
        return DiscreteSpec(len(ACTIONS))

    def reset(self, *, seed=None, options=None) -> Dict[str, Any]:
        if options:
            raise ValueError('GridGameEnvironment has no reset options')
        obs = self.env.reset(seed=seed)
        self._step_num = 0
        self.total_reward = 0.
        self.solved = False
        self.optimal_steps = self.env._manhattan()
        return {'observation': obs, 'instruction': obs['instruction']}

    def step(self, outputs) -> Tuple[Dict[str, Any], Optional[str]]:
        action = self.action_spec().validate(outputs['motor'])
        obs, reward, done, info = self.env.step(action)
        self._step_num += 1
        self.total_reward += reward
        self.solved = bool(info['reached_goal'])
        row = {'observation': obs, 'instruction': obs['instruction'], 'reward': reward}
        if self.solved:
            return row, 'terminal'
        return row, 'truncated' if (done or self._step_num >= self.max_steps) else None


def greedy_oracle_policy(row: Mapping[str, Any]) -> Dict[str, int]:
    """Move to reduce Manhattan distance to the goal (reads privileged state).

    A policy: ``row -> {'motor': action}``; it reads ``row['observation']``.
    Optimal on wall-free boards; a strong (not always optimal) baseline with
    walls. Used as the upper-reference in the closed-loop validation.
    """
    st = row['observation']['_state']
    (ar, ac), (gr, gc) = st['agent'], st['goal']
    walls = set(map(tuple, st['walls']))
    size = st['size']
    candidates = []
    for a, (dr, dc) in _DELTA.items():
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in walls:
            candidates.append((abs(nr - gr) + abs(nc - gc), a))
    return {'motor': min(candidates)[1] if candidates else 0}


def random_action_policy(seed: int = 0) -> Callable:
    """Return a uniform-random discrete policy (``row -> {'motor': action}``), the closed-loop null floor."""
    return _seeded_random_policy(seed, lambda row: len(ACTIONS))


def _seeded_random_policy(seed: int, n_actions: Callable) -> Callable:
    """A uniform-random policy whose draws repeat from run to run.

    One generator, seeded once, draws the actions of every step and episode.
    ``reset()`` runs when an episode begins; the first view of the episode then
    sets the generator to the state it had when that view first began an
    episode. The first run over a set of episodes draws one continuous stream;
    a later run over the same episodes, by the same subject or by a new one
    with the same seed, draws the same actions.
    """
    rng = np.random.RandomState(seed)
    starts, begun = {}, [False]

    def policy(row) -> Dict[str, int]:
        if begun[0]:
            begun[0] = False
            view = np.asarray(row['observation']['frame']).tobytes()
            rng.set_state(starts.setdefault(view, rng.get_state()))
        return {'motor': int(rng.randint(0, n_actions(row)))}

    policy.reset = lambda: begun.__setitem__(0, True)
    return policy


def play_game(subject, env: GridGameEnv, max_steps: Optional[int] = None,
              *, seed=None) -> Dict[str, Any]:
    """Run one grid episode through an EnvironmentSource; return the usual score.

    The caller owns subject reset. SessionProtocol provides isolated trials when
    using tools or comparing conditions.
    """
    wrapper = GridGameEnvironment(env, max_steps=max_steps)
    responses = run_environment(subject, wrapper, seed=seed, require_specs=True)
    actions = [int(np.asarray(action).item()) for action in responses]
    taken = len(actions)
    return {
        'solved': wrapper.solved,
        'steps': taken,
        'total_reward': round(wrapper.total_reward, 4),
        'optimal_steps': wrapper.optimal_steps,
        'efficiency': round(wrapper.optimal_steps / taken, 3) if (wrapper.solved and taken) else 0.0,
        'actions': actions,
    }


def evaluate_policy(make_model: Callable[[Callable], Any],
                    policy: Callable,
                    n_episodes: int = 20,
                    size: int = 6,
                    n_walls: int = 0,
                    base_seed: int = 100) -> Dict[str, float]:
    """Play ``n_episodes`` fresh games with ``policy`` and aggregate.

    ``make_model(policy)`` builds the ``BrainScoreModel`` wrapping the policy
    (kept as a factory so the caller controls model identity/wrappers). Returns
    success rate, mean steps-to-solve, and mean efficiency.
    """
    solves, steps, effs = [], [], []
    for i in range(n_episodes):
        env = GridGameEnv(size=size, n_walls=n_walls, seed=base_seed + i)
        model = make_model(policy)
        res = play_game(model, env)
        solves.append(1.0 if res['solved'] else 0.0)
        if res['solved']:
            steps.append(res['steps'])
            effs.append(res['efficiency'])
    return {
        'success_rate': float(np.mean(solves)),
        'mean_steps_to_solve': float(np.mean(steps)) if steps else float('nan'),
        'mean_efficiency': float(np.mean(effs)) if effs else 0.0,
        'n_episodes': n_episodes,
    }
