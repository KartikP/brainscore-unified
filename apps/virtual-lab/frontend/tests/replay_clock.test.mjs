import assert from 'node:assert/strict';
import test from 'node:test';
import {followAudio,replayTime} from '../dist/replay_clock.js';

function clock(time) {
  let current=time,seeks=0;
  return {duration:3,readyState:4,seeking:false,paused:false,ended:false,
    get currentTime(){return current;},set currentTime(value){current=value;seeks++;},get seeks(){return seeks;}};
}

test('a persistent small clock offset never seeks the audible stream',()=>{
  const video=clock(1.08),audio=clock(1);
  let last=-Infinity;
  for(let frame=0;frame<180;frame++)last=followAudio(video,audio,frame*16,last);
  assert.equal(audio.seeks,0); assert.equal(video.seeks,0);
  assert.equal(replayTime(video,audio),1);
});

test('large drift corrects only silent video, with a cooldown and no concurrent seek',()=>{
  const video=clock(.5),audio=clock(1);
  let last=followAudio(video,audio,0,-Infinity);
  assert.equal(video.currentTime,1);assert.equal(video.seeks,1);assert.equal(audio.seeks,0);
  video.currentTime=.5;const before=video.seeks;
  assert.equal(followAudio(video,audio,100,last),last);assert.equal(video.seeks,before);
  video.seeking=true;assert.equal(followAudio(video,audio,600,last),last);assert.equal(video.seeks,before);
  video.seeking=false;last=followAudio(video,audio,600,last);
  assert.equal(last,600);assert.equal(audio.seeks,0);
});

test('video clock continues after a shorter soundtrack ends or cannot advance',()=>{
  const video=clock(2),audio=clock(1);
  audio.ended=true;assert.equal(replayTime(video,audio),2);
  audio.ended=false;audio.paused=true;assert.equal(replayTime(video,audio),2);
  audio.paused=false;audio.readyState=1;assert.equal(replayTime(video,audio),2);
  assert.equal(replayTime(video),2);
});
