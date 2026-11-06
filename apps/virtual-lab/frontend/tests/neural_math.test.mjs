import assert from 'node:assert/strict';
import test from 'node:test';
import { difference, largestChanges, variableUnits, distance, checkComparisonSize, loadComparisons } from '../dist/neural_math.js';
const vector = values => Float64Array.from(values);

test('pairwise differences remove persistent outliers, retain signs and leave missing values missing', () => {
  const a = vector([-1000, 3, -2, NaN]), b = vector([-1000, 1, 2, 3]);
  const delta = difference(a, b);
  assert.deepEqual([...delta], [0, 2, -4, NaN]);
  assert.deepEqual(largestChanges(delta), [2, 1, 0]);
  assert.deepEqual([...a], [-1000, 3, -2, NaN]);
  assert.throws(() => difference(a, vector([1])), /matching/);
});

test('variation ranks changing units rather than units with large constant offsets', () => {
  const rows = [vector([1000, 0, 1, NaN]), vector([1000, 3, 1, NaN]), vector([1000, 6, 2, 4])];
  const { means, indices } = variableUnits(rows);
  assert.deepEqual(indices, [1, 2, 0]);
  assert.deepEqual([...means], [1000, 3, 4 / 3, 4]);
  assert.equal(rows[0][0] - means[0], 0);
  assert.equal(rows[0][1] - means[1], -3);
});

test('distances distinguish magnitude changes from correlation and report their finite-unit denominator', () => {
  const a = vector([1, 2, 3, NaN]), b = vector([2, 4, 6, 100]);
  assert.deepEqual(distance(a, b, 'rms'), { value: Math.sqrt(14 / 3), count: 3 });
  assert.deepEqual(distance(a, b, 'correlation'), { value: 0, count: 3 });
  assert.deepEqual(distance(vector([1, 2, 3]), vector([3, 2, 1]), 'correlation'), { value: 2, count: 3 });
  assert.deepEqual(distance(vector([1, 1]), vector([2, 2]), 'correlation'), { value: null, count: 2 });
  assert.deepEqual(distance(vector([NaN]), vector([1]), 'rms'), { value: null, count: 0 });
  assert.throws(() => distance(a, vector([1]), 'rms'), /matching/);
});

test('large comparisons are explicitly refused before allocating or silently sampling', () => {
  checkComparisonSize(3, 2048);
  assert.throws(() => checkComparisonSize(1, 10), /at least two/);
  assert.throws(() => checkComparisonSize(65, 10), /budget/);
  assert.throws(() => checkComparisonSize(2, 65537), /budget/);
  assert.throws(() => checkComparisonSize(64, 20000), /budget/);
});

test('paged comparison loader includes the final neuron of every stimulus and preserves the fixed slice', async () => {
  const calls = [];
  const get = async (presentation, offset) => {
    calls.push([presentation, offset]);
    return { presentation, offset, total_rows: 3, total_columns: 2051, stimulus: `s${presentation}`,
      values: Array.from({ length: Math.min(2048, 2051 - offset) }, (_, index) => presentation * 10000 + offset + index), slice: { time_bin: 0 } };
  };
  const result = await loadComparisons(get, new AbortController().signal);
  assert.equal(calls.length, 6);
  assert.deepEqual(result.labels, ['s0', 's1', 's2']);
  assert.deepEqual(result.rows.map(row => row[2050]), [2050, 12050, 22050]);
  assert.deepEqual(result.slice, { time_bin: 0 });
});

test('loader rejects inconsistent and oversized recordings and cooperatively cancels', async () => {
  const page = { presentation: 0, offset: 0, total_rows: 2, total_columns: 2, stimulus: 'a', values: [1, 2], slice: {} };
  await assert.rejects(loadComparisons(async () => ({ ...page, total_rows: 65 }), new AbortController().signal), /budget/);
  await assert.rejects(loadComparisons(async () => page, new AbortController().signal), /Inconsistent/);
  const stop = new AbortController(); stop.abort();
  await assert.rejects(loadComparisons(async () => page, stop.signal), /cancelled/);
});
