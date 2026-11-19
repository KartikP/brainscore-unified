import { binAtTime, checkReplayBudget, loadReplayChannel, type ReplayChannel, type ReplayManifest } from './replay_math.js';
import { timeLabel } from './temporal_math.js';
import { followAudio, mediaDuration, replayTime, ReplayTransport } from './replay_clock.js';

const node = <K extends keyof HTMLElementTagNameMap>(tag: K, text = '', className = '') => {
  const element = document.createElement(tag); element.textContent = text; element.className = className; return element;
};
const number = (value: number) => Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : 'missing';
interface RegionTile { update: (milliseconds: number) => void }

function regionTile(channel: ReplayChannel, values: Float64Array | undefined, extent: number): { element: HTMLElement; tile: RegionTile } {
  const element = node('section', '', 'replay-region'), heading = node('h3', channel.channel.replace(/^neural:/, ''));
  element.dataset.channel = channel.channel;
  const status = node('p', '', 'replay-bin'), canvas = node('canvas'); canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
  const scroll = node('div', '', 'replay-region-scroll'); scroll.append(canvas);
  const detail = node('p', '', 'plot-selection'); detail.setAttribute('aria-live', 'polite');
  const legend = node('div', '', 'activation-legend');
  legend.append(node('span', `-${number(extent)}`), node('span', '', 'activation-ramp'), node('span', `+${number(extent)}`));
  const fixed = Object.entries(channel.slice).map(([name, index]) => `${name}=${index}`).join(', ');
  element.append(heading, status, scroll, legend, detail, node('p', `${channel.units.toLocaleString()} neurons in recorded order.${fixed ? ` Fixed slice: ${fixed}.` : ''}`, 'subtle'));
  const columns = Math.ceil(Math.sqrt(channel.units)), rows = Math.ceil(channel.units / columns);
  const cell = Math.max(1, Math.min(12, Math.floor(280 / columns)));
  canvas.width = columns; canvas.height = rows; canvas.style.width = `${columns * cell}px`; canvas.style.height = `${rows * cell}px`;
  const context = canvas.getContext('2d')!;
  let lastBin: number | undefined, selected = 0;
  const inspect = () => {
    const value = lastBin !== undefined && lastBin >= 0 && values ? values[lastBin * channel.units + selected] : NaN;
    detail.textContent = `Neuron index ${selected}: ${number(value)}`;
  };
  const update = (milliseconds: number) => {
    const bin = values ? channel.time_bins.length ? binAtTime(channel.time_bins, milliseconds) : 0 : -3;
    if (bin === lastBin) return;
    lastBin = bin;
    status.textContent = bin === -3 ? 'No saved response for this stimulus' : bin === -2 ? 'Overlapping time bins; use Results to inspect them' : bin < 0
      ? 'No recorded bin at this time, or physical time unavailable' : channel.time_bins.length ? timeLabel(channel.time_bins[bin]) : 'Static recorded response';
    const pixels = context.createImageData(columns, rows);
    for (let index = 0; index < columns * rows; index++) {
      const value = bin >= 0 && values ? values[bin * channel.units + index] : NaN;
      const missing = index >= channel.units || !Number.isFinite(value), target = value < 0 ? [72, 111, 157] : [181, 119, 58];
      const strength = missing ? 0 : Math.min(1, Math.abs(value) / extent);
      for (let rgb = 0; rgb < 3; rgb++) pixels.data[index * 4 + rgb] = missing ? 220 : Math.round(250 + (target[rgb] - 250) * strength);
      pixels.data[index * 4 + 3] = index < channel.units ? 255 : 0;
    }
    context.putImageData(pixels, 0, 0);
    canvas.setAttribute('aria-label', `${channel.channel}: ${status.textContent}. All ${channel.units} neurons. Use arrow keys to inspect values.`);
    inspect();
  };
  canvas.onclick = event => {
    const index = Math.floor(event.offsetY / cell) * columns + Math.floor(event.offsetX / cell);
    if (index < channel.units) { selected = index; inspect(); }
  };
  canvas.onkeydown = event => {
    const offsets: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns, Home: -selected, End: channel.units - 1 - selected };
    if (event.key in offsets) { event.preventDefault(); selected = Math.max(0, Math.min(channel.units - 1, selected + offsets[event.key])); inspect(); }
  };
  return { element, tile: { update } };
}

export interface ReplayView { element: HTMLElement; pause: () => void; dispose: () => void }
export function replayView(runID: string): ReplayView {
  const element = node('div', '', 'replay-view'), controls = node('div', '', 'replay-controls');
  const selection = (label: string) => {
    const wrapper = node('label', label), select = node('select'); select.setAttribute('aria-label', `Replay ${label.toLowerCase()}`);
    wrapper.append(select); controls.append(wrapper); return select;
  };
  const condition = selection('Condition'), repetition = selection('Repetition'), stimulus = selection('Stimulus');
  const playback = node('div', '', 'replay-playback'), play = node('button', 'Play'); play.type = 'button'; play.disabled = true;
  play.setAttribute('aria-label', 'Play replay');
  const seek = node('input'); seek.type = 'range'; seek.min = '0'; seek.max = '0'; seek.step = '.001'; seek.value = '0'; seek.disabled = true;
  seek.setAttribute('aria-label', 'Replay time in seconds');
  const clock = node('output', '0.000 s'), speedLabel = node('label', 'Speed'), speed = node('select');
  speed.setAttribute('aria-label', 'Replay speed'); [.25, .5, 1, 2].forEach(value => speed.append(new Option(`${value}x`, String(value)))); speed.value = '1';
  speedLabel.append(speed); playback.append(play, seek, clock, speedLabel);
  const status = node('p', 'Read saved replay metadata...', 'subtle'); status.setAttribute('role', 'status');
  const media = node('div', '', 'replay-stimulus'), regions = node('div', '', 'replay-regions');
  element.append(controls, playback, status, media, regions,
    node('p', 'Saved activity only. Each region uses its native time bins and a separate color scale. Neuron positions follow recorded order. Timestamps locate output bins; models can use context across the clip.', 'subtle'));
  let manifest: ReplayManifest | undefined, tiles: RegionTile[] = [], master: HTMLMediaElement | undefined, audio: HTMLAudioElement | undefined;
  let controller: AbortController | undefined, epoch = 0, disposed = false, ready = false;
  let currentMs = 0, durationMs = 0, frame = 0, started = 0, startedMs = 0;
  let lastVideoCorrection = -Infinity;
  const extents = new Map<string, number>();

  const failed = (error: unknown) => {
    pause(); status.textContent = `Media playback unavailable. Use the time slider to inspect saved activity. ${String(error)}`;
  };
  let transport = new ReplayTransport(undefined, undefined, failed);
  const pause = () => {
    transport.pause(); cancelAnimationFrame(frame);
    play.textContent = 'Play'; play.setAttribute('aria-label', 'Play replay');
  };
  const releaseMedia = () => {
    pause();
    for (const item of [master, audio]) if (item) {
      item.onended = item.onerror = item.onloadedmetadata = null;
      item.removeAttribute('src'); item.load();
    }
    master = audio = undefined; media.replaceChildren(); transport = new ReplayTransport(undefined, undefined, failed);
  };
  const duration = () => {
    const saved = mediaDuration(master, audio);
    if (saved > 0) durationMs = saved * 1000;
    seek.max = String(durationMs / 1000); seek.disabled = !ready || durationMs <= 0;
    play.disabled = !ready || durationMs <= 0;
  };
  const draw = () => {
    seek.value = String(currentMs / 1000); clock.textContent = `${(currentMs / 1000).toFixed(3)} / ${(durationMs / 1000).toFixed(3)} s`;
    tiles.forEach(tile => tile.update(currentMs));
  };
  const tick = () => {
    if (!transport.playing || disposed) return;
    currentMs = master ? replayTime(master, audio) * 1000 : Math.min(durationMs, startedMs + (performance.now() - started) * Number(speed.value));
    if (master) lastVideoCorrection = followAudio(master, audio, performance.now(), lastVideoCorrection);
    draw();
    if (!master && currentMs >= durationMs) pause(); else frame = requestAnimationFrame(tick);
  };
  // The view starts at once; it does not wait for play() promises (WebKit can leave them pending).
  const begin = () => {
    if (!ready || play.disabled || disposed) return;
    currentMs = transport.play(currentMs / 1000, durationMs / 1000, Number(speed.value)) * 1000;
    if (!transport.playing) return;
    started = performance.now(); startedMs = currentMs;
    play.textContent = 'Pause'; play.setAttribute('aria-label', 'Pause replay');
    cancelAnimationFrame(frame); tick();
  };
  play.onclick = () => { if (transport.playing) pause(); else begin(); };
  seek.oninput = () => {
    currentMs = Number(seek.value) * 1000; started = performance.now(); startedMs = currentMs;
    transport.seek(currentMs / 1000);
    lastVideoCorrection = performance.now();
    draw();
  };
  speed.onchange = () => {
    started = performance.now(); startedMs = currentMs;
    if (master) master.playbackRate = Number(speed.value);
    if (audio) audio.playbackRate = Number(speed.value);
  };
  const get = async (url: string, signal: AbortSignal) => {
    const response = await fetch(url, { signal }), value = await response.json();
    if (!response.ok) throw new Error(value.error ?? 'Saved replay data unavailable'); return value;
  };

  const loadSelection = async () => {
    if (!manifest) return;
    releaseMedia(); controller?.abort(); controller = new AbortController(); const signal = controller.signal, token = ++epoch;
    ready = false; durationMs = currentMs = 0; tiles = []; regions.replaceChildren(); duration(); draw();
    lastVideoCorrection = -Infinity;
    status.textContent = 'Read saved neural activity...';
    const saved = manifest, position = Number(stimulus.value), selected = saved.stimuli[position];
    try {
      checkReplayBudget(saved.channels);
      if (!selected) throw new Error('Choose a saved stimulus');
      media.append(node('h3', selected.id));
      if (selected.text) media.append(node('pre', selected.text, 'replay-text'));
      const videoAsset = selected.media.find(item => item.kind === 'video'), audioAsset = selected.media.find(item => item.kind === 'audio');
      const imageAsset = selected.media.find(item => item.kind === 'image');
      if (imageAsset?.url) { const image = node('img'); image.src = imageAsset.url; image.alt = selected.id; media.append(image); }
      if (videoAsset?.url) {
        const video = node('video'); video.src = videoAsset.url; video.preload = 'metadata'; video.muted = true; video.playsInline = true;
        video.setAttribute('aria-label', 'Saved video stimulus'); media.append(video); master = video;
      }
      if (audioAsset?.url) {
        const sound = node('audio'); sound.src = audioAsset.url; sound.preload = 'auto'; sound.setAttribute('aria-label', 'Saved model audio'); media.append(sound);
        if (master) audio = sound; else master = sound;
        media.append(node('p', 'Audio: saved model input.', 'subtle'));
      }
      for (const asset of selected.media) if (asset.error) media.append(node('p', `${asset.kind}: ${asset.error}`, 'error'));
      if (selected.audio_preparation) media.append(node('p', selected.audio_preparation, 'subtle'));
      transport = new ReplayTransport(master, audio, failed);
      // Replay ends when every stream has ended: a shorter video must not cut longer audio.
      const ended = () => { if (transport.ended()) { currentMs = durationMs; pause(); draw(); } };
      for (const item of [master, audio]) if (item) {
        item.onloadedmetadata = () => { duration(); draw(); };
        item.onended = ended;
      }
      if (master) master.onerror = () => { pause(); status.textContent = 'The browser cannot decode this saved media. The time slider can still inspect saved neural bins.'; };
      for (const channel of saved.channels) {
        channel.time_bins.forEach(bin => { if (bin.end_ms !== null && Number.isFinite(bin.end_ms)) durationMs = Math.max(durationMs, bin.end_ms); });
        const presentation = channel.presentations[position];
        let values: Float64Array | undefined, extent = extents.get(channel.channel) ?? 0;
        if (presentation !== null) {
          const source = `/api/runs/${encodeURIComponent(runID)}/activation?index=${saved.index}&channel=${encodeURIComponent(channel.channel)}`;
          values = await loadReplayChannel(source, channel, presentation, url => get(url, signal), signal,
            loaded => { if (token === epoch) status.textContent = `Read saved ${channel.channel}: ${loaded.toLocaleString()} values`; });
          for (const value of values) if (Number.isFinite(value)) extent = Math.max(extent, Math.abs(value));
        }
        if (token !== epoch || disposed) return;
        extent ||= 1; extents.set(channel.channel, extent);
        const region = regionTile(channel, values, extent); regions.append(region.element); tiles.push(region.tile);
      }
      if (token !== epoch || disposed) return;
      ready = true; duration(); draw(); element.dataset.resultIndex = String(saved.index);
      status.textContent = `${saved.channels.length} saved neural regions. Select Play or move the time slider. No new model run.`;
      if (durationMs <= 0) status.textContent = 'Saved static responses. No media duration or physical neural clock is available for Play.';
    } catch (error) { if (token === epoch && !signal.aborted) status.textContent = String(error); }
  };
  const loadManifest = async () => {
    releaseMedia(); controller?.abort(); controller = new AbortController(); const signal = controller.signal, token = ++epoch;
    ready = false; duration(); status.textContent = 'Read saved replay metadata...';
    const conditionIndex = Number(condition.value || 0), repetitionIndex = Number(repetition.value || 0), stimulusIndex = Number(stimulus.value || 0);
    try {
      const saved = await get(`/api/runs/${encodeURIComponent(runID)}/replay?condition=${conditionIndex}&repetition=${repetitionIndex}`, signal) as ReplayManifest;
      if (token !== epoch || disposed) return;
      manifest = saved;
      condition.replaceChildren(...saved.conditions.map((name, index) => new Option(name, String(index)))); condition.value = String(saved.condition);
      repetition.replaceChildren(...Array.from({ length: saved.repetitions }, (_, index) => new Option(String(index + 1), String(index)))); repetition.value = String(saved.repetition);
      stimulus.replaceChildren(...saved.stimuli.map((item, index) => new Option(`${index + 1}: ${item.id}`, String(index)))); stimulus.value = String(Math.min(stimulusIndex, saved.stimuli.length - 1));
      await loadSelection();
    } catch (error) { if (token === epoch && !signal.aborted) status.textContent = String(error); }
  };
  condition.onchange = repetition.onchange = () => void loadManifest(); stimulus.onchange = () => void loadSelection();
  const visibility = () => { if (document.hidden) pause(); }; document.addEventListener('visibilitychange', visibility);
  queueMicrotask(() => { if (element.isConnected) void loadManifest(); });
  return { element, pause, dispose: () => { disposed = true; controller?.abort(); releaseMedia(); document.removeEventListener('visibilitychange', visibility); } };
}
