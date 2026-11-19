// Replay transport: R1 (pending or rejected play() promises), R2 (unequal durations), R3 (audio not cut at video end).
import assert from 'node:assert/strict';
import test from 'node:test';
const {ReplayTransport, mediaDuration, followAudio} = await import('../dist/replay_clock.js');

// Fake media. mode: 'resolve' | 'late' | 'never' | 'reject'. pause() rejects a pending play like browsers do.
function media(duration, mode = 'resolve') {
  const m = {duration, currentTime: 0, readyState: 4, seeking: false, paused: true, ended: false, playbackRate: 1,
    mode, plays: 0, pending: [], seeks: 0};
  let t = 0;
  Object.defineProperty(m, 'currentTime', {get: () => t, set: v => {t = Math.min(v, duration); m.seeks++; m.ended = false;}});
  m.play = () => {
    m.plays++; if (m.ended) {t = 0; m.ended = false;} m.paused = false;
    if (m.mode === 'reject') { m.paused = true; return Promise.reject(new Error('NotAllowedError')); }
    if (m.mode === 'resolve') return Promise.resolve();
    return new Promise((resolve, reject) => m.pending.push({resolve, reject}));
  };
  m.pause = () => { m.paused = true; for (const p of m.pending.splice(0)) p.reject(new Error('AbortError')); };
  m.advance = s => { if (m.paused) return; t = Math.min(duration, t + s); if (t >= duration) { m.ended = true; m.paused = true; } };
  m.settle = () => { for (const p of m.pending.splice(0)) p.resolve(); };
  return m;
}
const flush = () => new Promise(r => setTimeout(r, 0));
const setup = (videoS, audioS, mode) => {
  const video = media(videoS, mode), audio = media(audioS, mode), errors = [];
  return {video, audio, errors, transport: new ReplayTransport(video, audio, e => errors.push(e))};
};

test('R1: end -> Play -> Pause -> Play is playing at once while play() never settles', async () => {
  const {video, audio, errors, transport} = setup(6, 6, 'never');
  transport.play(0, 6, 1); video.advance(6); audio.advance(6);
  assert.equal(transport.ended(), true); assert.equal(transport.playing, false);
  assert.equal(transport.play(6, 6, 1), 0); assert.equal(transport.playing, true);
  assert.equal(video.currentTime, 0); assert.equal(audio.currentTime, 0);
  transport.pause(); await flush();
  assert.equal(transport.playing, false); assert(audio.paused && video.paused); assert.equal(errors.length, 0);
  transport.play(1, 6, 1); await flush();
  assert.equal(transport.playing, true); assert.equal(errors.length, 0); assert(!audio.paused && !video.paused);
});

test('R1: seek -> Pause -> Play with a late promise', async () => {
  const {audio, errors, transport} = setup(6, 6, 'late');
  transport.play(0, 6, 1); assert.equal(transport.playing, true);
  transport.seek(2); assert.equal(audio.currentTime, 2);
  transport.pause(); transport.play(2, 6, 1); assert.equal(transport.playing, true);
  audio.settle(); await flush();
  assert.equal(transport.playing, true); assert.equal(errors.length, 0);
});

test('R1: Pause during a pending play wins over a later resolution or rejection', async () => {
  const {video, audio, errors, transport} = setup(6, 6, 'late');
  transport.play(0, 6, 1); transport.pause();
  video.settle(); audio.settle(); await flush();
  assert.equal(transport.playing, false); assert(audio.paused && video.paused); assert.equal(errors.length, 0);
});

test('R1: rejection of the current request rolls back once, with the error', async () => {
  const {audio, errors, transport} = setup(6, 6, 'reject');
  transport.play(0, 6, 1); assert.equal(transport.playing, true);
  await flush();
  assert.equal(transport.playing, false); assert.equal(errors.length, 1); assert(audio.paused);
});

test('R2: longer audio continues after video end; timeline is the longest stream', async () => {
  const {video, audio, transport} = setup(3, 8);
  assert.equal(mediaDuration(video, audio), 8);
  transport.play(0, 8, 1); video.advance(3); audio.advance(3);
  assert.equal(transport.ended(), false); assert(!audio.paused && transport.playing);
  transport.pause(); assert.equal(transport.play(4, 8, 1), 4);
  assert.equal(video.plays, 1, 'ended video is not restarted'); assert(!audio.paused);
  assert.equal(followAudio(video, audio, 1e6, -Infinity), -Infinity, 'no seeks of an ended video');
  audio.advance(8); assert.equal(transport.ended(), true); assert.equal(transport.playing, false);
});

test('R2: shorter audio lets video continue; repeat resets both to zero', async () => {
  const {video, audio, transport} = setup(8, 3);
  assert.equal(mediaDuration(video, audio), 8);
  transport.play(0, 8, 1); video.advance(3); audio.advance(3);
  assert.equal(transport.ended(), false); assert(!video.paused);
  video.advance(5); assert.equal(transport.ended(), true);
  assert.equal(transport.play(8, 8, 1), 0); assert.equal(video.currentTime, 0); assert.equal(audio.currentTime, 0);
});

test('R3: equal durations, video ends first; audio is not cut', () => {
  const {video, audio, transport} = setup(6, 6);
  transport.play(0, 6, 1); video.advance(6); audio.advance(5.93);
  assert.equal(transport.ended(), false); assert(!audio.paused);
  audio.advance(1); assert.equal(transport.ended(), true);
});
