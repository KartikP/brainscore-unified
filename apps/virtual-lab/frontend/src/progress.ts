export interface RunProgress {
  phase: string;
  condition?: string;
  repetition?: number | string;
  activity?: string;
  recording_neural?: boolean;
  progress?: { completed: number; total: number; unit: string };
}

export function progressView(run: RunProgress | null, state: string, connected = true) {
  const active = Boolean(run && ['queued', 'running', 'trial_finished', 'cancelling'].includes(run.phase));
  const completed = run?.phase === 'complete';
  let activity = !connected ? 'Connection lost; experiment status unavailable' : state === 'loading' ? 'Loading subject' :
    state === 'validating' ? 'Checking protocol compatibility' : run?.phase === 'failed' ? 'Experiment failed' :
    run?.phase === 'cancelled' ? 'Experiment stopped' : completed ? 'Experiment complete' :
    active ? run?.phase === 'queued' ? 'Preparing experiment' : run?.activity ?? 'Processing experiment' : 'Ready for an experiment';
  if (connected && run?.phase === 'cancelling') activity += ' · stop requested';
  const progress = run?.progress;
  const known = progress && progress.unit === 'repetition' && Number.isInteger(progress.total) && progress.total > 0 &&
    Number.isInteger(progress.completed) && progress.completed >= 0 && progress.completed <= progress.total;
  const value = known ? completed ? progress.total : progress.completed : completed ? 1 : undefined;
  const total = known ? progress.total : 1;
  const details: string[] = [];
  if (known) details.push(`${value} of ${total} condition repetitions complete`);
  if (active && run?.condition) details.push(run.condition);
  if (active && run?.repetition !== undefined) details.push(`repetition ${typeof run.repetition === 'number' ? run.repetition + 1 : run.repetition}`);
  if (active && run?.recording_neural) details.push('neural recording requested');
  if (active) details.push(run?.condition === 'benchmark' || run?.condition === 'loading_benchmark'
    ? 'This benchmark does not expose intermediate progress.'
    : 'Progress updates after each complete repetition; a model may process stimuli as one batch.');
  return { visible: active || Boolean(run) || state === 'loading' || state === 'validating',
    activity, detail: details.join(' · '), value, total, showBar: active || completed || Boolean(known) };
}

export function renderProgress(run: RunProgress | null, state: string, connected: boolean): void {
  const view = progressView(run, state, connected);
  document.getElementById('lab-progress')!.hidden = !view.visible;
  document.getElementById('lab-activity')!.textContent = view.activity;
  document.getElementById('lab-progress-detail')!.textContent = view.detail;
  const bar = document.getElementById('experiment-progress') as HTMLProgressElement;
  bar.hidden = !view.showBar;
  bar.max = view.total;
  if (view.value === undefined) bar.removeAttribute('value');
  else bar.value = view.value;
}
