# A new kind of output

Your experiment needs something from the subject that no other guide shows. This page says which way to take, and shows one closed-loop example from start to end.

## Which way

| The subject gives... | Way | Who names it | Who checks the payload | Who produces it | Who reads it |
| --- | --- | --- | --- | --- | --- |
| An answer to a trial: a choice, probabilities, a rating, a response time, text, log probabilities | One of the six answer formats, a closed list ([Behavior](behavior.md)). Nothing to add. | The benchmark, in `TaskContext(response=...)` | Core, by the format's schema | The model's answer, reply or readout part | The benchmark, which compares it with human answers |
| An action that changes what it sees next: a key press, a pointer move, a motor command, "wait" | A closed loop ([Closed loop](closed_loop.md)). The action goes on `motor`, or on another channel the policy names. | The policy (`policy_out_channels`); the benchmark requests it | The session, with the environment's `action_spec()` | The policy, once per step | The environment (`step`), then the benchmark |
| A measurement that is neither, for example eye position | A new registered channel ([Extend UMI](extend.md#a-new-channel)) | You, with `register_channel` | The session, with the channel's `shape_validator` | A model part (a policy can return it at each step) or your own subject | The benchmark |

## A channel or a capability

A channel names *what* flows in a session and what its payload looks like. A capability is *how* `BrainScoreModel` runs a session that requests certain outputs. `BrainScoreModel` has four built-in capabilities: neural, behavior, motor and intervention sessions.

| | Channel | Capability |
| --- | --- | --- |
| Says | what a payload is | how `BrainScoreModel` produces it |
| Registered with | `register_channel` (core) | `register_capability` (unified) |
| Needed by | every subject that sends the output | `BrainScoreModel` only |

A new kind of output needs a new capability only when no part of `BrainScoreModel` can produce it. A policy can return any registered output channel in its `policy_out_channels` (the model refuses another name when it is built), so an action or a per-step measurement needs no capability. A subject that is not a `BrainScoreModel` implements `participate(session)` and needs no capability at all.

## Example: a screen task

This task is made up; it has no human data. Each step shows the subject a picture of a screen, with text drawn on it. The subject moves the pointer, clicks, presses a key, waits, or runs the simulation one tick. The screen says "WAIT" until two ticks have run, then "CLICK" and the button turns green. A click on the green button ends the episode. The environment states the form of an action with the specification classes of core. One mapping with every field is the smallest form the session checks completely: a step that does not use `x`, `y` or `key` sends 0.

```python
import numpy as np
from PIL import Image, ImageDraw
from brainscore import BrainScoreModel
from brainscore_core.environment import DiscreteSpec, MappingSpec

KINDS = ('move', 'click', 'key', 'wait', 'run')
KEYS = ('enter', 'escape', 'space')
SIZE = 64

class ScreenTask:
    def action_spec(self):
        return MappingSpec({'kind': DiscreteSpec(len(KINDS)), 'x': DiscreteSpec(SIZE),
                            'y': DiscreteSpec(SIZE), 'key': DiscreteSpec(len(KEYS))})

    def reset(self, seed=None):
        self.ticks, self.steps = 0, 0
        return {'observation': self.screen()}

    def screen(self):
        ready = self.ticks >= 2
        picture = Image.new('RGB', (SIZE, SIZE), 'white')
        draw = ImageDraw.Draw(picture)
        draw.text((2, 2), 'CLICK' if ready else 'WAIT', fill='black')
        draw.rectangle((40, 40, 56, 56), fill='green' if ready else 'gray')   # the button
        return np.asarray(picture)

    def step(self, outputs):
        action = outputs['motor']            # already checked against action_spec()
        kind = KINDS[action['kind']]
        self.steps, self.ticks = self.steps + 1, self.ticks + (kind == 'run')
        on_button = 40 <= action['x'] <= 56 and 40 <= action['y'] <= 56
        if kind == 'click' and on_button and self.ticks >= 2:
            return {'observation': self.screen()}, 'terminal'
        return {'observation': self.screen()}, ('truncated' if self.steps >= 10 else None)
```

The policy is a plain function. It reads the picture and returns one action:

```python
def action(kind, x=0, y=0, key=0):
    return {'kind': KINDS.index(kind), 'x': x, 'y': y, 'key': key}

def policy(row):
    red, green, blue = row['observation'][48, 48]   # the centre of the button
    return {'motor': action('click', 48, 48) if green > red else action('run')}

subject = BrainScoreModel(identifier='screen-agent', policy=policy,
                        policy_in_channels=('observation',), policy_out_channels=('motor',))
```

Run one episode. The actions and the reason the episode ended come back from the session:

```python
from brainscore_core.streaming_helpers import environment_session

with environment_session(ScreenTask()) as session:
    subject.participate(session)
actions = [event.payload for event in session.collect('motor')]
reason = session.source.end_reason   # 'terminal', 'truncated' or 'failure'; None while it runs
subject.reset()
print([KINDS[step['kind']] for step in actions], reason)   # ['run', 'run', 'click'] terminal
```

- **The session takes one action per step.** A list of actions in one step fails a mapping specification ("Expected a mapping"). Without an `action_spec()` the environment receives the list unchecked. A sequence of actions is the episode.
- **It checks each action** against `action_spec()` before `step` runs. A failed check raises `ValueError` with the field and the reason, for example `x: Action exceeds declared bounds; no implicit wrapping` for `action('click', 99)`, and the episode ends as `failure`.
- **The policy receives the picture as the environment returned it**: here an `(H, W, 3)` `uint8` array. Put the picture under `observation`, as the grid game does, and add `observation_spec()`, for example `ArraySpec((SIZE, SIZE, 3), 'uint8')`, to have the session check it; the specification checks only that key. A path stays a path; the session does not load it.

**Scoring.** A benchmark runs the episodes and compares what the subject did with what people did on the same screens: the sequence of actions, or the outcome (solved, number of steps). The grid game benchmark (`GridGame-reach-5x5`, `brainscore/benchmarks/grid_game/`) is the complete registered closed-loop example; it scores task success against an oracle and a random floor, not against human data. `brainscore.harnesses.human_replay.HumanAgreementBenchmark` shows the human comparison: it replays the states people saw and scores agreement with their choices. See [Write a benchmark](write_a_benchmark.md) for the declarations.

**A model that reads pictures and writes text.** `api_policy` (`brainscore.model_helpers.api_behavioral`) makes a hosted model a policy, but it answers with one integer from a menu of legal actions: give the environment a `DiscreteSpec` over a fixed list of actions to use it. For a local vision-language model there is no policy helper. The reply part of `chat_parts` takes a `Prompt`, so a few lines turn it into a policy:

```python
import io
from brainscore.model_helpers.replies import chat_parts
from brainscore_core.responses import Prompt, Text

reply, _ = chat_parts(vlm, processor)   # a Hugging Face vision-language model and its processor
ASK = 'Answer with four numbers: the kind (0 move, 1 click, 2 key, 3 wait, 4 run), x, y, key.'

def vlm_policy(row):
    screen = io.BytesIO()
    Image.fromarray(row['observation']).save(screen, format='PNG')
    prompt = Prompt(parts=(('text', 'Click the button when it is green.'), ('image', screen), ('text', ASK)),
                    names=('instruction', 'screen', 'directive'), response=Text(max_tokens=16))
    numbers = [int(word) for word in reply(prompt).split() if word.isdigit()]
    return {'motor': dict(zip(('kind', 'x', 'y', 'key'), numbers))}
```

**Limits.** A model that answers in a form the specification refuses ends the episode as `failure`. `MappingSpec` has no form for "one of several kinds, each with its own fields"; every action carries every field. `HumanAgreementBenchmark` compares integer actions; a structured action needs an integer code first.
