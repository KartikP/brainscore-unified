import assert from 'node:assert/strict';
import test from 'node:test';
import { progressView } from '../dist/progress.js';

test('progress stays tied to completed repetitions across all conditions', () => {
  const run = { phase: 'running', condition: 'baseline', repetition: 1,
    activity: 'Viewing stimuli and generating responses', recording_neural: true,
    progress: { completed: 1, total: 6, unit: 'repetition' } };
  const view = progressView(run, 'ready');
  assert.equal(view.value, 1);
  assert.equal(view.total, 6);
  assert.match(view.detail, /1 of 6 condition repetitions complete/);
  assert.match(view.detail, /repetition 2/);
  assert.match(view.detail, /neural recording requested/);
  assert.equal(progressView({ ...run, phase: 'complete' }, 'ready').value, 6);
  assert.equal(progressView({ ...run, phase: 'complete' }, 'ready').activity, 'Experiment complete');
  assert.equal(progressView({ ...run, phase: 'failed' }, 'ready').value, 1);
  assert.equal(progressView({ ...run, phase: 'cancelled' }, 'ready').activity, 'Experiment stopped');
  assert.match(progressView({ ...run, phase: 'cancelling' }, 'ready').activity, /stop requested/);
  assert.match(progressView(run, 'ready', false).activity, /status unavailable/);
});

test('unknown benchmark progress is indeterminate and old runs remain readable', () => {
  const view = progressView({ phase: 'running', condition: 'benchmark', activity: 'Running benchmark' }, 'ready');
  assert.equal(view.value, undefined);
  assert.match(view.detail, /does not expose intermediate progress/);
  assert.equal(progressView({ phase: 'queued' }, 'ready').value, undefined);
  assert.equal(progressView({ phase: 'complete' }, 'ready').value, 1);
  assert.equal(progressView(null, 'ready').visible, false);
});
