import assert from 'node:assert/strict';
import test from 'node:test';
import { barDomain, answerCounts, activationExtent } from '../dist/plots.js';

test('bar domains include zero without clipping negative or missing values', () => {
  assert.deepEqual(barDomain([-.75, .25, null, NaN]), [-.75, .25]);
  assert.deepEqual(barDomain([.2, .8]), [0, .8]);
  assert.deepEqual(barDomain([0, null]), [0, 1]);
  assert.deepEqual(barDomain([null, Infinity]), [0, 1]);
});

test('response counts preserve exact raw answers, including parse errors', () => {
  const rows = [{ raw_answer: 'Yes' }, { raw_answer: 'yes' }, { raw_answer: 'Yes', valid: false }, { raw_answer: '' }, {}];
  assert.deepEqual(answerCounts(rows), [{ label: 'Yes', value: 2 }, { label: 'yes', value: 1 }, { label: '', value: 1 }]);
  assert.equal(answerCounts(rows).reduce((sum, item) => sum + item.value, 0), 4);
});

test('all activation conditions share one symmetric scale without treating missing cells as zero', () => {
  assert.equal(activationExtent([{ values: [[-8, 2], [null, NaN]] }, { values: [[0, .1]] }]), 8);
  assert.equal(activationExtent([{ values: [[0, null]] }]), 1);
  assert.equal(activationExtent([]), 1);
});
