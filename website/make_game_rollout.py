"""Render the static game-rollout filmstrip shown above the playable game.

Pure matplotlib (no model, no brainscore import) — just draws the 5x5 grid for
one of Qwen2.5-VL-7B's recorded successful rollouts (seed 504, an optimal solve),
frame by frame, matching the env's colours. Light-mode friendly (white panels).

Board + action trace come straight from model_moves.js (seed 504):
    player=[1,2] goal=[2,4] actions=[3,1,3]  (right, down, right) -> optimal 3.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SIZE = 5
BG = (235/255, 235/255, 235/255)
PLAYER = (40/255, 90/255, 220/255)
GOAL = (40/255, 190/255, 70/255)
GRID = (0.78, 0.78, 0.78)
DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
NAMES = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

player0 = [1, 2]
goal = [2, 4]
actions = [3, 1, 3]

# build the sequence of player positions
positions = [player0[:]]
p = player0[:]
for a in actions:
    dr, dc = DELTA[a]
    p = [p[0] + dr, p[1] + dc]
    positions.append(p[:])

n_frames = len(positions)
fig, axes = plt.subplots(1, n_frames, figsize=(3.0 * n_frames, 3.4))
fig.patch.set_alpha(0.0)  # transparent figure bg; panels drawn white below


def draw(ax, player, title):
    ax.set_facecolor('white')
    ax.add_patch(Rectangle((0, 0), SIZE, SIZE, facecolor=BG, edgecolor='none', zorder=0))
    for i in range(SIZE + 1):
        ax.plot([i, i], [0, SIZE], color=GRID, lw=1, zorder=1)
        ax.plot([0, SIZE], [i, i], color=GRID, lw=1, zorder=1)
    # goal (row r -> y from top), draw with row 0 at top
    pad = 0.12
    gr, gc = goal
    ax.add_patch(Rectangle((gc + pad, SIZE - 1 - gr + pad), 1 - 2 * pad, 1 - 2 * pad,
                           facecolor=GOAL, edgecolor='none', zorder=2))
    pr, pc = player
    ax.add_patch(Rectangle((pc + pad, SIZE - 1 - pr + pad), 1 - 2 * pad, 1 - 2 * pad,
                           facecolor=PLAYER, edgecolor='none', zorder=3))
    ax.set_xlim(0, SIZE); ax.set_ylim(0, SIZE)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title, fontsize=12, color='#1b2333', pad=8)


titles = ['start']
for a in actions:
    titles.append(f'→ {NAMES[a]}')
titles[-1] += '  ✓ goal'

for ax, pos, title in zip(axes, positions, titles):
    draw(ax, pos, title)

fig.suptitle('Qwen2.5-VL-7B (chain-of-thought) — one optimal rollout (seed 504, 3/3 moves)',
             fontsize=13, color='#1b2333', y=1.02)
fig.tight_layout()
fig.savefig('assets/game_rollout.png', dpi=140, bbox_inches='tight', transparent=True)
print('wrote assets/game_rollout.png')
