import assert from 'node:assert/strict';
import test from 'node:test';
import { dataURL, timeLabel, physicalIntervals, traceSegments, neuronIndices } from '../dist/temporal_math.js';

const bin = (index, start_ms = null, end_ms = null) => ({ index, start_ms, end_ms, center_ms: null });

test('data requests retain channel and repetition, replace the bin, and use the temporal endpoint', () => {
  const source = '/api/runs/fixture/activation?index=3&channel=neural%3AA1&time_bin=0';
  const snapshot = new URL(dataURL(source, { time_bin: 7, limit: 1 }), 'http://lab.local');
  assert.equal(snapshot.searchParams.get('channel'), 'neural:A1');
  assert.deepEqual(snapshot.searchParams.getAll('time_bin'), ['7']);
  assert.equal(snapshot.searchParams.get('index'), '3');
  const temporal = new URL(dataURL('/api/runs/fixture/activation?channel=neural%3AIT', { time_offset: 5 }, true), 'http://lab.local');
  assert.equal(temporal.pathname, '/api/runs/fixture/temporal');
  assert.equal(temporal.searchParams.get('time_offset'), '5');
});

test('time labels use recorded physical bounds and never convert sample indices into milliseconds', () => {
  assert.equal(timeLabel(bin(3, 80, 100)), 'Bin 4: 80 to 100 ms');
  assert.equal(timeLabel(bin(3)), 'Bin 4: physical time unavailable');
  assert.equal(physicalIntervals([bin(0, 0, 20), bin(1, 40, 60)]), true);
  assert.equal(physicalIntervals([bin(0, 20, 0)]), false);
  assert.equal(physicalIntervals([bin(0)]), false);
});

test('traces preserve physical intervals and break at genuine gaps and missing responses', () => {
  const bins = [bin(0, 0, 20), bin(1, 20, 40), bin(2, 60, 80), bin(3, 80, 100), bin(4, 100, 120)];
  const segments = traceSegments(bins, [1, 3, 2, null, -1]);
  assert.deepEqual(segments, [
    [{ x: 0, y: 1 }, { x: 20, y: 1 }, { x: 20, y: 3 }, { x: 40, y: 3 }],
    [{ x: 60, y: 2 }, { x: 80, y: 2 }], [{ x: 100, y: -1 }, { x: 120, y: -1 }],
  ]);
  assert.deepEqual(traceSegments([bin(2), bin(3), bin(4)], [1, NaN, 3]), [[{ x: 3, y: 1 }], [{ x: 5, y: 3 }]]);
  assert.throws(() => traceSegments(bins, [1]), /equal lengths/);
});

test('selected trace neurons retain explicit order and reject duplicates, invalid or oversized selections', () => {
  assert.deepEqual(neuronIndices('7, 0, 19'), [7, 0, 19]);
  for (const text of ['', '0,0', '-1', '1.5', 'a', '0,1,2,3,4,5,6,7,8', '9007199254740992'])
    assert.throws(() => neuronIndices(text));
});
