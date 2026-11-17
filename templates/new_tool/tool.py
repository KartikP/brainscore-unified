"""TODO: one-line description of what your tool observes or changes.

A tool is equipment attached to an experiment. This one watches the outputs of a run and
keeps, for each recorded region, the mean response over stimuli and units.

**This file runs as-is.** `region_means(subject, stimuli, regions)` runs a passive
experiment with the tool and returns `{region: mean}`. Replace `on_event` with your own
analysis.
"""
import numpy as np

from brainscore.experiments import Experiment, TaskProtocol, Tool
from brainscore_core import TaskContext


class YourTool(Tool):
    """The mean response of each recorded region."""

    name = 'your_tool'

    def __init__(self, regions=('IT',)):
        # The outputs this tool asks the subject for. TaskProtocol requests them, so the
        # run sends a `neural:<region>` output for each region.
        self.requested_output_channels = frozenset(f'neural:{region}' for region in regions)
        self.means = {}

    def on_event(self, event):
        # `event` is a dict: event['kind'] says what happened; for 'input' and 'output'
        # event['payload'] is a StreamEvent (channel, payload, t_ms, meta). Never modify a
        # payload; copy what you keep.
        if event['kind'] != 'output' or not event['payload'].channel.startswith('neural:'):
            return
        region = event['payload'].channel.split(':', 1)[1]
        self.means[region] = float(np.asarray(event['payload'].payload.values, dtype=float).mean())

    def describe(self):
        return {'regions': sorted(channel.split(':', 1)[1]
                                  for channel in self.requested_output_channels)}


def region_means(subject, stimuli, regions=('IT',)) -> dict:
    """Show `stimuli` once to `subject` with the tool attached; return `{region: mean}`."""
    tool = YourTool(regions)
    protocol = TaskProtocol(stimuli, TaskContext(task_type='passive'), identifier='region-means',
                            conditions=['intact'], repetitions=[0])
    Experiment(subject=subject, protocol=protocol, tools=[tool]).run()
    return dict(tool.means)
