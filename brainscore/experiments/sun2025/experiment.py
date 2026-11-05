"""A direction-and-speed judgment through the ordinary UMI task protocol."""
import json
import math
import re
from contextlib import contextmanager
from dataclasses import replace

from brainscore_core import TaskContext
from brainscore_core.responses import Text
from brainscore_core.streaming import BufferedSession
from brainscore.experiments import TaskProtocol

INSTRUCTION = (
    'Watch the short video and judge the motion of the texture or distorted region '
    'at the center, around the time the central dot flashes. Judge the moving region, '
    'not the flashing dot. Report the perceived direction and speed. '
    'Direction is in degrees counterclockwise from right: right=0, up=90, left=180, down=270. '
    'Speed is in pixels per video frame, measured in the ORIGINAL 600 by 600 pixel clip. '
    'It has 16 frames at 30 frames per second. A motion traversing the clip width in '
    'one second would be 20 pixels per frame. Report zero speed only for no perceived motion.'
)
DIRECTIVE = ('Reply with only a JSON object containing two numeric fields: '
             '"direction_deg" and "speed_px_per_frame". No explanation or Markdown.')
REPORT_PARSER_VERSION = 'motion-json-v2'


def report_format(reply):
    """Recognize a whole-response JSON code fence, never text surrounding an answer."""
    fence = chr(96) * 3
    match = re.fullmatch(r'\s*' + fence + r'json\s*\n(.*?)\n' + fence + r'\s*',
                        reply, re.DOTALL) if isinstance(reply, str) else None
    return (match.group(1), 'json_code_fence') if match else (reply, 'plain_json')


class _ClipSession(BufferedSession):
    """Address the visual form explicitly while preserving the normal trial metadata."""
    def next_input(self):
        event = super().next_input()
        if event is not None and event.channel == 'vision':
            return replace(event, channel='vision:clip', t_ms=None)
        return event


class ClipTaskProtocol(TaskProtocol):
    """A TaskProtocol whose declared and emitted visual input is a native clip."""
    def __init__(self, stimuli, task, **kwargs):
        super().__init__(stimuli, task, **kwargs)
        if self.input_channels != frozenset({'vision'}):
            raise ValueError('Sun motion stimuli must contain only the clip input stream')
        self.input_channels = frozenset({'vision:clip'})

    def validate(self, subject):
        if 'vision:clip' not in subject.in_channels:
            raise ValueError(f'{subject.identifier!r} does not declare native vision:clip input')
        super().validate(subject)

    @contextmanager
    def _session(self, repetition):
        yield _ClipSession(stimulus_set=self.stimuli, task_context=self.task,
                           requested_output_channels=sorted(self.output_channels))


def parse_report(reply: str) -> tuple[dict | None, str | None]:
    """Accept one plain/fenced JSON object; never infer or change numerical values."""
    def unique_fields(pairs):
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError('duplicate JSON field')
        return dict(pairs)

    try:
        body, _ = report_format(reply)
        value = json.loads(body, object_pairs_hook=unique_fields)
    except (TypeError, ValueError):
        return None, 'expected a JSON object'
    if not isinstance(value, dict) or set(value) != {'direction_deg', 'speed_px_per_frame'}:
        return None, 'expected exactly direction_deg and speed_px_per_frame'
    for name, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            return None, f'{name} must be a finite number'
    if not 0 <= value['direction_deg'] < 360:
        return None, 'direction_deg must be in [0, 360)'
    if value['speed_px_per_frame'] < 0:
        return None, 'speed_px_per_frame must be nonnegative'
    return value, None


def make_protocol(stimuli, *, seed: int = 0, repetitions=(0,)):
    """Every row is one clip trial; protocol repetitions own session resets."""
    ordered = stimuli.sample(frac=1, random_state=seed).reset_index(drop=True)
    ordered.identifier = f'{stimuli.identifier}-order-{seed}'
    ordered.stimulus_paths = dict(stimuli.stimulus_paths)
    return ClipTaskProtocol(
        ordered, TaskContext(instruction=INSTRUCTION + '\n\n' + DIRECTIVE,
                             response=Text(), present=('video',)),
        identifier=f'Sun2025-motion-report-v1-{ordered.identifier}',
        repetitions=repetitions,
        metadata={'study_doi': '10.1038/s42256-025-01068-w', 'shuffle_seed': seed,
                  'response_adaptation': 'numerical report replaces human motion-matching adjustment',
                  'display_adaptation': 'versioned raster target interval; no calibrated monitor or matching loop',
                  'speed_units': 'displayed pixels per frame', 'memory': 'independent clip judgments'})
