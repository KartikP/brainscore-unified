"""Regenerate saved streaming figures using the notebook's own UMI setup.

Run from any directory in the coordinated notebook environment. The script runs
all code cells in notebook 15, then plots their measured delivery/timing values.
It does not maintain a second copy of the experiment logic.
"""
from pathlib import Path
import hashlib
import json
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOTEBOOK = Path(__file__).resolve().parents[1] / '15_streaming_delivery.ipynb'
ASSETS = NOTEBOOK.parent / 'assets' / 'streaming'
ASSETS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlepad': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})
notebook = nbformat.read(NOTEBOOK, as_version=4)
code = '\n\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
exec(compile(code, str(NOTEBOOK), 'exec'))

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))

# Each block spans the inputs included in one call.
ax = axes[0]
delivery = [
    ('One at a time', stream_subject.call_sizes),
    ('Batch', batch_subject.call_sizes),
]
for row, (label, sizes) in enumerate(delivery):
    start = 0
    for size in sizes:
        ax.broken_barh(
            [(start + 0.05, size - 0.1)],
            (row + 0.6, 0.8),
            color=['#0072B2', '#D55E00'][row],
        )
        start += size
ax.set_yticks([1, 2], ['One at a time: 6 calls', 'Batch: 1 call'])
ax.set_xlabel('Input position')
ax.set_title('Inputs per model call')

# Window times and sizes come from the session's metadata.
ax = axes[1]
for window in short_windows:
    start = window['window_start_ms']
    width = window['window_end_ms'] - start
    ax.broken_barh(
        [(start, width)],
        (0.6, 0.8),
        color='#D55E00' if window['partial'] else '#0072B2',
        edgecolor='white',
    )
    label = f"{window['n_frames']} frames"
    if window['partial']:
        label += '\npartial'
    ax.text(
        start + width / 2,
        1,
        label,
        ha='center',
        va='center',
        color='white',
        fontsize=8,
    )
ax.set_yticks([])
ax.set_xlabel('Experiment time (ms)')
ax.set_title('75 frames in one-second windows')


# Finish the figure with the long feed's input-window sizes.
ax = axes[2]
window_sizes = [window['n_frames'] for window in long_windows]
window_numbers = np.arange(1, len(window_sizes) + 1)
ax.plot(
    window_numbers,
    np.cumsum(window_sizes),
    color='0.45',
    label='Frames received so far',
)
ax.plot(
    window_numbers,
    window_sizes,
    color='#0072B2',
    label='Frames per input window',
)
ax.set_xlabel('Window number')
ax.set_ylabel('Frames')
ax.set_title(f'Maximum input buffer: {long_max} frames')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ASSETS / 'input_delivery.png', dpi=180, bbox_inches='tight')
plt.close(fig)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(11, 4))
for row, (spans, report, error, finished) in enumerate(runs):
    y = len(runs) - row - 1
    ax.plot(range(10), [y + 0.4] * 10, '|', color='0.45', ms=12)
    for index, start, end in spans:
        ax.broken_barh(
            [(start, end - start)],
            (y + 0.05, 0.7),
            color='#0072B2',
            edgecolor='white',
        )
        if end - start >= 1:
            ax.text(
                (start + end) / 2,
                y + 0.4,
                str(index),
                ha='center', va='center', color='white', fontsize=8,
            )
    if error is None:
        delivered = {span[0] for span in spans}
        skipped = sorted(set(range(10)) - delivered)
        assert len(skipped) == report['windows_dropped']
        ax.plot(skipped, [y + 0.4] * len(skipped), 'x', color='#D55E00')
    else:
        ax.text(finished + 0.5, y + 0.4, 'Stopped: lag exceeded', va='center')
ax.axvline(10, color='0.35', linestyle='--')
ax.set_yticks([3.4, 2.4, 1.4, 0.4], [setting[0] for setting in settings])
ax.set_xlabel('Simulated clock time (seconds)')
ax.set_title('One feed, different responses to late inputs')
ax.legend(handles=[
    Patch(color='#0072B2', label='Processing'),
    Line2D([], [], color='0.45', marker='|', linestyle='', label='Scheduled start'),
    Line2D([], [], color='#D55E00', marker='x', linestyle='', label='Skipped'),
    Line2D([], [], color='0.35', linestyle='--', label='Feed ends'),
], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
fig.tight_layout()
fig.savefig(ASSETS / 'clock_policies.png', dpi=180, bbox_inches='tight')
plt.close(fig)

metadata = {
    'source': NOTEBOOK.name,
    'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
    'evidence': 'Executed deterministic session demonstration; simulated clock.',
    'maximum_frames_held': {'75_frame_feed': short_max, '600_frame_feed': long_max},
    'timing': summary,
}
(ASSETS / 'measurements.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Saved figures:', ASSETS)
