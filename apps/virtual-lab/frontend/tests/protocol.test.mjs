import assert from 'node:assert/strict';
import test from 'node:test';
import { responseChannels, requestedRecordings } from '../dist/protocol.js';

test('task response and recorders are independent decisions', () => {
  const task = { kind: 'choice', probabilities: true, confidence: { low: 1, high: 5, step: 1 } };
  assert.deepEqual(responseChannels(task), ['behavior:choice', 'behavior:probabilities', 'behavior:confidence']);
  assert.deepEqual(requestedRecordings({ task, measurements: { behavior: false, regions: ['IT', 'V4'] } }), ['neural:IT', 'neural:V4']);
  assert.deepEqual(requestedRecordings({ task, measurements: { behavior: true, regions: ['IT'] } }), ['neural:IT', 'behavior:choice', 'behavior:probabilities', 'behavior:confidence']);
});

test('plain viewing, typed text, ratings and legacy behavior keep their channel meanings', () => {
  assert.deepEqual(responseChannels({ kind: 'passive' }), []);
  assert.deepEqual(responseChannels({ kind: 'text' }), ['behavior:text']);
  assert.deepEqual(responseChannels({ kind: 'rating' }), ['behavior:rating']);
  assert.deepEqual(responseChannels({ kind: 'legacy' }), ['behavior']);
});
