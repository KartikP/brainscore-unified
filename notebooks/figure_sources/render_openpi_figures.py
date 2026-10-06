"""Render measured trained-policy results from the bundled experiment records."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from brainscore.run_record import RunRecord

ASSETS = Path(__file__).resolve().parents[1] / 'assets/openpi'
COLORS = {'baseline': '#0072B2', 'ablation': '#D55E00', 'restored': '#009E73'}


def first_call_activity(route, source):
    rows = [event for event in RunRecord(ASSETS / 'records' / route).events()
            if event['kind'] == 'activity' and event['source'] == source]
    first_id = rows[0]['event_id']
    return [event['payload']['value'] for event in rows if event['event_id'] == first_id]


def first_actions(route):
    return next(event['payload']['actions']
                for event in RunRecord(ASSETS / 'records' / route).events()
                if event['kind'] == 'output')


def main():
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    before = first_call_activity('ablation', 'before')
    after = first_call_activity('ablation', 'after')
    request = next(event['payload']['args'][0]
                   for event in RunRecord(ASSETS / 'records/baseline').events()
                   if event['kind'] == 'input')
    x = np.arange(len(before))
    fig, (camera, ax) = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    camera.imshow(request['observation/image'])
    camera.set_title('What the policy received')
    camera.axis('off')
    for offset, values, label, color in [
        (-0.18, before, 'Before silencing', COLORS['baseline']),
        (0.18, after, 'After silencing', COLORS['ablation']),
    ]:
        heights = [float(np.abs(value['array']).mean()) for value in values]
        bars = ax.bar(x + offset, heights, width=0.36, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=4)
    ax.set_xticks(x, [str(value['denoising_step']) for value in before])
    ax.set_xlabel('Internal denoising step (not simulator time)')
    ax.set_ylabel('Mean absolute value of selected unit')
    ax.set_title('The selected internal value becomes zero')
    ax.margins(y=0.4)
    ax.legend(loc='upper right', fontsize=9)
    fig.suptitle('Record and silence an internal action update\nTrained pi05_libero · NVIDIA L4 · action_out_proj unit 0')
    fig.savefig(ASSETS / 'activity.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained', sharey=True)
    for index, ax in enumerate(axes):
        for route, label, style in [
            ('baseline', 'Baseline', '-'),
            ('ablation', 'Unit 0 silenced at steps 0 and 1', '-'),
            ('restored', 'After tool removal', '--'),
        ]:
            values = first_actions(route)[:, index]
            ax.plot(np.arange(len(values)) + 1, values, style,
                    color=COLORS[route], marker='o', label=label,
                    markerfacecolor='none' if route == 'restored' else COLORS[route],
                    markersize=9 if route == 'restored' else 5,
                    linewidth=1.5 if route == 'restored' else 2.5)
        ax.set_title(f'Movement command value {index}')
        ax.set_xlabel('Position in predicted action chunk')
        ax.set_xticks([1, 5, len(values)])
        ax.axhline(0, color='0.7', linewidth=0.7, zorder=0)
    axes[0].set_ylabel('Predicted value (LIBERO controller scale)')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('An internal intervention changes predicted actions\nTrained pi05_libero · same observation and starting random state')
    fig.savefig(ASSETS / 'actions.png', dpi=160)
    plt.close(fig)
    np.testing.assert_array_equal(first_actions('baseline'), first_actions('recording'))
    np.testing.assert_array_equal(first_actions('baseline'), first_actions('restored'))
    assert all(np.all(value['array'] == 0) for value in after)


if __name__ == '__main__':
    main()
