"""The plan of one session: decided once when it starts, before any compute.

``BrainScoreModel.participate`` builds it with ``plan_session`` and keeps it as
``model._session_plan`` while the session runs. The session parts and the
answering code read it instead of deciding again (Contract Rules 4.5, 5.14):
which capability runs the session, the pass plan of
``BRAINSCORE_REGIONS_PER_PASS``, how the trials are delivered, and the part
that answers. A combination that cannot run is refused here with
``ExecutionNotFeasibleError``, before any forward pass.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.streaming import parse_channel
from brainscore_core.streaming_helpers import refuse_variance, session_outputs


@dataclass(frozen=True)
class SessionPlan:
    """What one session runs, decided at its start."""

    handler: object                     # the capability that runs the session
    regions_per_pass: str               # 'all' or 'one' (BRAINSCORE_REGIONS_PER_PASS)
    answers: Tuple[str, ...]            # the requested behavior channels, in request order
    regions: Tuple[str, ...]            # the requested regions, in request order
    task_path: bool                     # the session drives its task (answers, or regions during it)
    delivery: str                       # 'table', 'per_trial' (one trial at a time) or 'stream'
    pick: Optional[Tuple[str, Optional[str]]] = None    # (answering part, probability source)
    pick_key: Optional[tuple] = None    # the task and formats the pick was made for
    by_answer: Optional[bool] = None    # within generation: the answer part (True) or the reply part
    recordings: Optional[tuple] = None  # groups that record the regions during the answering pass or a step
    passive: Optional[tuple] = None     # without a task: (region order, shared calls, multi_modality)

    def pick_for(self, task, formats) -> Optional[Tuple[str, Optional[str]]]:
        """The planned pick when ``task`` and ``formats`` are the ones it was made for, else None."""
        if self.pick is None or _pick_key(task, formats) != self.pick_key:
            return None
        return self.pick


def _pick_key(task, formats) -> tuple:
    """Every input of ``BehavioralReadout._choose_path`` that comes from the session."""
    return (repr(task.response), task.instruction,
            None if task.label_set is None else tuple(task.label_set),
            task.fitting_stimuli is None,
            None if task.present is None else tuple(task.present),
            task.task_type, tuple(formats))


def plan_session(model, session, passes: str) -> SessionPlan:
    """The plan of ``session`` for ``model``; raises before any compute when it cannot run.

    ``session`` is the view of the session that the model receives (its declared
    inputs); ``passes`` is the pass plan ``participate`` read. The answering part
    is chosen here, once per session. Refusals: the outputs no one part answers
    together, a task session without a task, several time bins during a task, an
    answer format a legacy task does not give, trial-to-trial variance, a task no
    part can answer, trial history without a part that sees it, a region the
    readout cannot record from its input (``ExecutionNotFeasibleError``); a
    malformed task, a bad prompt layout and rows with different numbers of
    options (``ValueError``).
    """
    outputs = session_outputs(session)
    task = getattr(session, 'task', None)
    task_type = getattr(task, 'task_type', None) if task is not None else None
    try:
        handler = model.check_session_support(outputs)
    except ExecutionNotFeasibleError as error:
        raise ExecutionNotFeasibleError(
            f"Session refused (task type {task_type if task is not None else 'none'!r}): {error}",
            channels=error.channels, task_type=task_type) from error
    answers = tuple(c for c in outputs if parse_channel(c)[0] == 'behavior')
    regions = tuple(parse_channel(c)[1] for c in outputs if parse_channel(c)[0] == 'neural')
    streaming = bool(getattr(session, 'streaming', False))
    history = 'memory:trial_history' in model.traits

    from ..capabilities.sessions import _BEHAVIOR, _MOTOR, _NEURAL
    from .recording import plan_recordings
    if handler is _NEURAL:
        task_path = task is not None and task.task_type != 'passive'
    elif handler is _BEHAVIOR:
        task_path = set(outputs) != {'behavior'} or (task is not None and task.response is not None)
    else:
        task_path = False

    pick = pick_key = by_answer = recordings = passive = None
    if task_path:
        _refuse_task_session(model, session, task, answers, regions)
        delivery = 'stream' if streaming else 'per_trial' if history else 'table'
        if task.response is not None or task.task_type != 'passive':
            pick = _choose_answering_part(model, session, task, answers, regions, delivery, passes)
            pick_key = _pick_key(task, answers)
            if task.response is not None:
                by_answer = model._behavioral.by_answer(task)
            stimuli = _presented_table(session)
            if regions and pick[0] != 'readout' and stimuli is not None:
                # Regions recorded during the answering pass: refused, or grouped, before any fit.
                recordings = plan_recordings(model, stimuli, _region_layers(model, regions))
    elif handler is _NEURAL:
        refuse_variance(model, session)
        per_trial = history and (task is None or task.trial_memory == 'none')
        delivery = 'stream' if streaming else 'per_trial' if per_trial else 'table'
        stimuli = _presented_table(session)
        if stimuli is not None and regions:
            # Every region is checked before the first pass; the order and the shared calls
            # are decided once. The whole table reads several inputs at once (core's driver).
            from .channels import plan_channels
            multi = delivery == 'table' and _spans_several_inputs(stimuli)
            passive = plan_channels(model, stimuli, regions, getattr(session, 'time_bins', None),
                                    multi, passes) + (multi,)
    else:
        delivery = 'stream' if streaming else 'table'
        if handler is _MOTOR and regions:
            # Regions recorded during each policy step: refused, or grouped, before the first step.
            recordings = plan_recordings(model, None, _region_layers(model, regions))
        if handler is _BEHAVIOR and task is not None and task.task_type != 'passive':
            # Bare behavior of a legacy task (core's typed driver): its part is chosen here too.
            refuse_variance(model, session)
            model._behavioral.validate_task(task)
            pick = model._behavioral._choose_path(task, ['behavior'])
            pick_key = _pick_key(task, ['behavior'])
    return SessionPlan(handler=handler, regions_per_pass=passes, answers=answers, regions=regions,
                       task_path=task_path, delivery=delivery, pick=pick, pick_key=pick_key,
                       by_answer=by_answer, recordings=recordings, passive=passive)


def _presented_table(session):
    """The session's table when its source presents exactly that table's rows, else None.

    Only then can the plan check the whole table, group the recordings and read its
    input forms before any set-up: ``stimulus_session`` states it (core
    ``Session._presents_table``, listed and streaming). A source given with a table
    by its caller may present other rows or other forms, so each batch decides, as
    before the plan.
    """
    return session.stimuli if getattr(session, '_presents_table', False) else None


def _spans_several_inputs(stimuli) -> bool:
    """Whether the table's stimulus columns feed more than one channel family (as core's driver reads its events)."""
    from brainscore_core.io_catalog import column_channel
    channels = {column_channel(column) for column in stimuli.columns} - {None}
    return len({parse_channel(channel)[0] for channel in channels}) > 1


def _region_layers(model, regions) -> dict:
    """``{region: its layer paths}``, as the session parts build it."""
    from .recording import region_selector
    return {region: list(region_selector(model, region).layer_paths) for region in regions}


def _refuse_task_session(model, session, task, answers, regions) -> None:
    """The refusals of a session that drives its task, before any setup or compute."""
    outputs = set(answers) | {f'neural:{region}' for region in regions}
    if task is None:
        raise ExecutionNotFeasibleError(
            "A behavior session needs a task: build it with one, for example "
            "stimulus_session(stimuli, outputs, TaskContext(response=Choice([...]))).",
            channels=outputs)
    time_bins = getattr(session, 'time_bins', None)
    if time_bins is not None and len(time_bins) > 1:
        raise ExecutionNotFeasibleError(
            "Recording during a task returns one time bin; request at most one.",
            channels={f'neural:{region}' for region in regions} or outputs, task_type=task.task_type)
    formats = set(answers) - {'behavior'}
    if task.response is None and formats:
        raise ExecutionNotFeasibleError(
            f"{list(answers)} need TaskContext(response=...); task_type "
            f"{task.task_type!r} answers on 'behavior'.", channels=formats, task_type=task.task_type)
    # A format the task's response does not name was refused when the session was built (core).
    refuse_variance(model, session)


def _choose_answering_part(model, session, task, answers, regions, delivery, passes='all'):
    """The answering part and the probability source, chosen once; refused before any fit.

    The refusals that ``respond`` and the readout would raise after the readout
    was fitted are raised here: trial history without a reply or likelihood part,
    and, when the source presents exactly its table, a region the readout cannot record from
    the input it presents, a readout extraction (the readout layer and its regions) that
    no extractor of that input can make, a bad prompt layout and rows with different
    numbers of options (whole table only; one trial at a time or a stream compares
    nothing across trials). Any other source is checked per batch (a stated limit).
    """
    readout = model._behavioral
    readout.validate_task(task)
    pick = readout._choose_path(task, answers)
    path = pick[0]
    if (task.response is not None and path != 'readout' and model._memory == 'trial_history'
            and model._reply is None and model._likelihood is None):
        raise ExecutionNotFeasibleError(
            "memory='trial_history' needs reply= or likelihood=; answer= cannot see earlier trials.",
            channels=set(answers), task_type=task.task_type)
    stimuli = _presented_table(session)
    if stimuli is None:
        return pick                     # only the batches show their rows (rule 5.14, stated limit)
    # A presented table is checked whole, also for a streamed delivery;
    # options are compared across rows only when the whole table is answered at once.
    if path == 'readout':
        # The input the readout presents, then its whole extraction request, before the fit.
        modality = readout.refuse_readout_regions(stimuli, regions)
        readout.refuse_readout_request(stimuli, modality, _region_layers(model, regions), passes)
    if task.response is not None:
        from .behavioral import trial_prompts
        trial_prompts(task, stimuli, compare_options=delivery == 'table')
    return pick
