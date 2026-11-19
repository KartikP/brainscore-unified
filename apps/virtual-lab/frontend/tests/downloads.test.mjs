import assert from 'node:assert/strict';
import test from 'node:test';
import { downloadView } from '../dist/downloads.js';

test('download bars show real file progress and do not claim model completion', () => {
  const view = downloadView({ phase: 'downloading', file: 'weights.bin', downloaded_bytes: 500000, total_bytes: 1000000 }, 'loading');
  assert.equal(view.value, 500000);
  assert.equal(view.total, 1000000);
  assert.match(view.detail, /50% of this file/);
  assert.equal(downloadView({ phase: 'loading', file: 'weights.bin', downloaded_bytes: 100, total_bytes: 100 }, 'loading').label, 'Loading model');
  assert.equal(downloadView({ phase: 'downloading', downloaded_bytes: 100 }, 'loading').value, undefined);
  assert.equal(downloadView({ phase: 'failed' }, 'load_failed').label, 'Model load failed');
  assert.equal(downloadView(null, 'ready').visible, false);
});
