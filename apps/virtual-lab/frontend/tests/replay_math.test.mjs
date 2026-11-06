import assert from 'node:assert/strict';
import test from 'node:test';
import { binAtTime, checkReplayBudget, loadReplayChannel } from '../dist/replay_math.js';
const bins = (count, width) => Array.from({length:count}, (_, index) => ({index,start_ms:index*width,end_ms:(index+1)*width,center_ms:null}));
const channel = (units, count, width=20) => ({channel:'neural:A1',units,time_bins:bins(count,width),presentations:[0],slice:{}});

test('replay uses physical intervals independently for each channel and leaves gaps explicit', () => {
  assert.equal(binAtTime(bins(150, 20), 500), 25);
  assert.equal(binAtTime(bins(8, 375), 500), 1);
  assert.equal(binAtTime(bins(2, 20), 40), 1);
  assert.equal(binAtTime(bins(2, 20), 41), -1);
  const intervals = [bins(1,20)[0], {index:1,start_ms:40,end_ms:60,center_ms:null}];
  assert.equal(binAtTime(intervals,30), -1);
  assert.equal(binAtTime([{index:0,start_ms:null,end_ms:null,center_ms:null}],10), -1);
  assert.equal(binAtTime([{index:0,start_ms:0,end_ms:40}, {index:1,start_ms:20,end_ms:60}],30), -2);
});

test('replay budgets include all neurons and time bins across all regions before allocation', () => {
  checkReplayBudget([channel(768,150),channel(1024,8)]);
  assert.throws(() => checkReplayBudget([channel(4000,1001)]), /budget/);
  assert.throws(() => checkReplayBudget([channel(3000,1000),channel(2000,1000)]), /budget/);
  assert.throws(() => checkReplayBudget([channel(-1,10)]), /neuron/);
});

test('bulk replay loader includes the last bin and neuron and preserves missing values without model calls', async () => {
  const calls = [];
  const data = await loadReplayChannel('/api/runs/fixture/activation?index=3&channel=neural%3AA1',channel(2051,150),1,async source => {
    const url = new URL(source,'http://lab.local'); calls.push(url);
    assert.equal(url.pathname,'/api/runs/fixture/temporal');
    assert.equal(url.searchParams.get('index'),'3'); assert.equal(url.searchParams.get('labels'),'0');
    const offset=Number(url.searchParams.get('offset')),time=Number(url.searchParams.get('time_offset'));
    const limit=Number(url.searchParams.get('limit')),count=Number(url.searchParams.get('time_limit'));
    assert.ok(limit*count<=65536);
    return {total_columns:2051,total_time_bins:150,presentation:1,offset,time_offset:time,
      neurons:Array.from({length:limit},(_,index)=>offset+index),
      values:Array.from({length:count},(_,step)=>Array.from({length:limit},(_,unit)=>time+step===0&&offset+unit===0?null:(time+step)*10000+offset+unit))};
  },new AbortController().signal);
  assert.ok(Number.isNaN(data[0])); assert.equal(data[149*2051+2050],1492050);
  assert.equal(calls.length,7);
});

test('static replay outputs stay static and incomplete or cancelled pages fail explicitly', async () => {
  const output=channel(3,0), source='/api/runs/fixture/activation?index=0';
  const get=async()=>({total_columns:3,presentation:0,offset:0,values:[1,null,3]});
  assert.deepEqual([...await loadReplayChannel(source,output,0,get,new AbortController().signal)],[1,NaN,3]);
  await assert.rejects(loadReplayChannel(source,output,0,async()=>({total_columns:3,presentation:0,offset:0,values:[1]}),new AbortController().signal),/Incomplete/);
  const stop=new AbortController(); stop.abort();
  await assert.rejects(loadReplayChannel(source,output,0,get,stop.signal),/cancelled/);
});
