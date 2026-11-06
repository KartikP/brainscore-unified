export interface MediaClock {
  currentTime: number;
  duration: number;
  readyState: number;
  seeking: boolean;
  paused: boolean;
  ended: boolean;
}

export interface ReplayMedia extends MediaClock {
  playbackRate: number;
  play(): Promise<void> | void;
  pause(): void;
}

// Never seek the audible stream during steady playback. Decoder buffering can
// report a small persistent offset; repeatedly seeking it interrupts playback.
export function replayTime(video: MediaClock, audio?: MediaClock): number {
  return audio && !audio.ended && !audio.paused && audio.readyState >= 2
    ? audio.currentTime : video.currentTime;
}

export function followAudio(video: MediaClock, audio: MediaClock | undefined,
    now: number, lastCorrection: number): number {
  if (!audio || audio.ended || audio.paused || audio.seeking || video.seeking ||
      video.ended || video.paused || video.readyState < 2 || audio.readyState < 2 ||
      now - lastCorrection < 500)
    return lastCorrection;
  // Past the end of a shorter video there is no frame to follow.
  if (Number.isFinite(video.duration) && audio.currentTime >= video.duration) return lastCorrection;
  if (Math.abs(video.currentTime - audio.currentTime) > .2) {
    video.currentTime = audio.currentTime;
    return now;
  }
  return lastCorrection;
}

const known = (seconds: number) => Number.isFinite(seconds) && seconds > 0;
const covers = (stream: MediaClock, seconds: number) => !known(stream.duration) || seconds < stream.duration;
const atEnd = (stream: MediaClock) => stream.ended || known(stream.duration) && stream.currentTime >= stream.duration;

// The replay timeline lasts until the longest saved stream ends.
export function mediaDuration(...streams: (MediaClock | undefined)[]): number {
  return Math.max(0, ...streams.map(stream => stream && known(stream.duration) ? stream.duration : 0));
}

// Playback state does not wait for play() promises. WebKit can leave a play()
// promise pending after the media has started, so the view starts immediately
// and a rejection of the current request rolls it back. Pause always wins: it
// invalidates every pending request, so a late resolution or rejection is ignored.
export class ReplayTransport {
  playing = false;
  private request = 0;
  constructor(readonly master: ReplayMedia | undefined, readonly audio: ReplayMedia | undefined,
      private readonly failed: (error: unknown) => void) {}

  private get streams(): ReplayMedia[] {
    return [this.master, this.audio].filter((stream): stream is ReplayMedia => stream !== undefined);
  }

  // Start at positionSeconds and return the start position. Play after the end starts at zero.
  play(positionSeconds: number, durationSeconds: number, rate: number): number {
    const request = ++this.request, streams = this.streams;
    const restart = positionSeconds >= durationSeconds || streams.length > 0 && streams.every(atEnd);
    const start = restart ? 0 : positionSeconds;
    for (const stream of streams) stream.playbackRate = rate;
    if (restart && this.master) this.master.currentTime = 0;
    // The audible stream is the clock, so an explicit start positions it.
    if (this.audio && this.audio.readyState > 0) this.audio.currentTime = Math.min(start, this.audio.duration || Infinity);
    this.playing = true;
    this.resume(streams.filter(stream => covers(stream, start)), request);
    return start;
  }

  // Move every stream to positionSeconds; while playing, restart streams that cover it.
  seek(positionSeconds: number): void {
    for (const stream of this.streams)
      if (stream.readyState > 0) stream.currentTime = Math.min(positionSeconds, stream.duration || Infinity);
    if (this.playing)
      this.resume(this.streams.filter(stream => stream.paused && covers(stream, positionSeconds)), this.request);
  }

  pause(): void {
    this.request++;
    this.playing = false;
    for (const stream of this.streams) stream.pause();
  }

  // Call on each 'ended' event. Returns true when no stream can still advance.
  ended(): boolean {
    if (!this.streams.every(stream => atEnd(stream) || stream.paused)) return false;
    this.pause();
    return true;
  }

  private resume(streams: ReplayMedia[], request: number): void {
    const fail = (error: unknown) => {
      if (request !== this.request || !this.playing) return;
      this.pause();
      this.failed(error);
    };
    for (const stream of streams) {
      try { void Promise.resolve(stream.play()).catch(fail); } catch (error) { fail(error); }
    }
  }
}
