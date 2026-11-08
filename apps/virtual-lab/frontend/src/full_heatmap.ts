import { dataURL, timeLabel, type TimeBin } from './temporal_math.js';

interface Page {
  total_rows: number;
  total_columns: number;
  presentation: number;
  stimulus: string;
  offset: number;
  values: (number | null)[];
  slice: Record<string, number>;
  extent?: number;
  labels?: string[];
  time?: TimeBin & { count: number };
}
const node = <K extends keyof HTMLElementTagNameMap>(tag: K, text = '') => {
  const element = document.createElement(tag); element.textContent = text; return element;
};
const format = (value: number) => Number.isInteger(value) ? String(value) : value.toPrecision(5);

export class HeatmapScale {
  private entries = new Map<HTMLElement, { extent: number; draw: (extent: number) => void }>();
  private seenExtent = 0;
  register(element: HTMLElement, draw: (extent: number) => void): (extent: number) => void {
    this.entries.set(element, { extent: 1, draw });
    return extent => {
      this.entries.get(element)!.extent = extent;
      this.seenExtent = Math.max(this.seenExtent, extent);
      for (const view of this.entries.keys()) if (!view.isConnected) this.entries.delete(view);
      const shared = Math.max(this.seenExtent, ...[...this.entries.values()].map(entry => entry.extent));
      for (const [view, entry] of this.entries) if (view.isConnected) entry.draw(shared);
    };
  }
}

export function fullHeatmap(source: string, scale: HeatmapScale, stimulusLabels: string[] = [],
    options: { presentation?: number; onPresentation?: (presentation: number) => void } = {}): HTMLElement {
  const container = node('div'); container.className = 'full-activation-plot';
  const controls = node('div'); controls.className = 'full-heatmap-controls';
  const stimulusLabel = node('label', 'Stimulus');
  const stimulus = node('input'); stimulus.type = 'number'; stimulus.min = '1'; stimulus.value = String((options.presentation ?? 0) + 1); stimulus.step = '1';
  stimulus.setAttribute('aria-label', 'Stimulus number for full activation heatmap'); stimulusLabel.append(stimulus);
  const zoomLabel = node('label', 'Zoom');
  const zoom = node('input'); zoom.type = 'range'; zoom.min = '100'; zoom.max = '400'; zoom.step = '25'; zoom.value = '100';
  zoom.setAttribute('aria-label', 'Activation heatmap zoom');
  const zoomValue = node('output', '100%'); zoomLabel.append(zoom, zoomValue);
  controls.append(stimulusLabel, zoomLabel);
  const status = node('p', 'Loading all recorded neurons…'); status.className = 'subtle'; status.setAttribute('role', 'status');
  const scroll = node('div'); scroll.className = 'full-heatmap-scroll';
  const surface = node('div'); surface.className = 'full-heatmap-surface';
  const canvas = node('canvas'); canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
  const selection = node('div'); selection.className = 'heatmap-selection'; selection.hidden = true;
  surface.append(canvas, selection); scroll.append(surface);
  const legend = node('div'); legend.className = 'activation-legend';
  const low = node('span'), high = node('span'), ramp = node('span'); ramp.className = 'activation-ramp';
  legend.append(low, ramp, high, node('span', '0 = neutral · gray = missing'));
  const detail = node('p', 'Click a neuron, or focus the heatmap and use arrow keys to inspect values.'); detail.className = 'plot-selection'; detail.setAttribute('aria-live', 'polite');
  const note = node('p', 'Every neuron is shown in recorded order, wrapped into rows. This is not an anatomical map. Colors show raw values on a shared scale; no averaging or normalization.'); note.className = 'subtle';
  container.append(controls, status, scroll, legend, detail, note);
  let values: Float64Array | null = null, columns = 1, rows = 1, cell = 1, selected = 0, extent = 1;
  let page: Page | undefined, controller: AbortController | undefined, sequence = 0;

  const layout = () => {
    const size = cell * Number(zoom.value) / 100;
    canvas.style.width = `${columns * size}px`; canvas.style.height = `${rows * size}px`;
    surface.style.width = canvas.style.width; surface.style.height = canvas.style.height;
    selection.style.left = `${selected % columns * size}px`; selection.style.top = `${Math.floor(selected / columns) * size}px`;
    selection.style.width = `${size}px`; selection.style.height = `${size}px`;
    zoomValue.textContent = `${zoom.value}%`;
  };
  const draw = (shared: number) => {
    extent = shared;
    low.textContent = `−${format(shared)}`; high.textContent = `+${format(shared)}`;
    if (!values) return;
    const context = canvas.getContext('2d')!;
    const pixels = context.createImageData(columns, rows);
    for (let index = 0; index < columns * rows; index++) {
      const value = values[index], missing = index >= values.length || !Number.isFinite(value);
      const color = value < 0 ? [72, 111, 157] : [181, 119, 58];
      const intensity = missing ? 0 : Math.min(1, Math.abs(value) / shared);
      for (let channel = 0; channel < 3; channel++) pixels.data[index * 4 + channel] = missing ? 220 : Math.round(250 + (color[channel] - 250) * intensity);
      pixels.data[index * 4 + 3] = index < values.length ? 255 : 0;
    }
    context.putImageData(pixels, 0, 0);
    canvas.setAttribute('aria-label', `All ${values.length} recorded neurons for stimulus ${page?.stimulus}. Raw activations; color range minus ${shared} to plus ${shared}. Use arrow keys to inspect neurons.`);
  };
  const setExtent = scale.register(container, draw);
  const inspect = async (index: number, focus = false) => {
    if (!values || !page) return;
    selected = Math.max(0, Math.min(values.length - 1, index)); selection.hidden = false; layout();
    const indexAtRequest = selected, presentation = page.presentation;
    const value = values[selected];
    const description = `Stimulus ${page.stimulus} / neuron ${selected}: ${Number.isFinite(value) ? format(value) : 'nonfinite / missing'}${page.time ? ` / ${timeLabel(page.time)}` : ''}`;
    detail.textContent = description;
    if (focus) selection.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    try {
      const requestSequence = sequence;
      const response = await fetch(dataURL(source, { presentation, offset: selected, limit: 1, labels: 1 }));
      if (!response.ok) return;
      const labeled = await response.json() as Page;
      if (requestSequence === sequence && selected === indexAtRequest && page.presentation === presentation && labeled.labels?.[0]) detail.textContent = `${description} (${labeled.labels[0]})`;
    } catch { /* The numeric value remains visible if its optional coordinate is unavailable. */ }
  };
  const load = async () => {
    controller?.abort(); controller = new AbortController();
    const current = ++sequence;
    const presentation = Number(stimulus.value) - 1;
    if (!Number.isInteger(presentation) || presentation < 0) { status.textContent = 'Enter a stimulus number starting at 1.'; return; }
    values = null; selection.hidden = true; status.textContent = 'Loading all recorded neurons…';
    const get = async (offset: number, stats = false): Promise<Page> => {
      const response = await fetch(dataURL(source, { presentation, offset, limit: 2048, stats: stats ? 1 : 0 }), { signal: controller!.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ?? 'Recorded activations unavailable');
      return data as Page;
    };
    try {
      let chunk = await get(0, true);
      if (current !== sequence || !container.isConnected) return;
      const first = chunk;
      // Keep the viewer's own numeric buffer below 128 MiB, independently of model memory.
      if (!Number.isSafeInteger(first.total_columns) || first.total_columns < 1 || first.total_columns > 16 * 1024 * 1024) throw new Error('This recording exceeds the interactive viewer memory budget; use the complete artifact export. No neurons have been silently omitted.');
      const buffer = new Float64Array(first.total_columns);
      let offset = 0;
      while (true) {
        if (chunk.offset !== offset || !chunk.values.length || offset + chunk.values.length > buffer.length) throw new Error('Inconsistent recorded activation page');
        chunk.values.forEach((value, index) => { buffer[offset + index] = value ?? NaN; });
        offset += chunk.values.length;
        status.textContent = `Loaded ${offset.toLocaleString()} / ${buffer.length.toLocaleString()} neurons`;
        if (offset === buffer.length) break;
        chunk = await get(offset);
        if (current !== sequence || !container.isConnected) return;
      }
      if (current !== sequence || !container.isConnected) return;
      page = { ...first, stimulus: stimulusLabels[presentation] ?? first.stimulus }; values = buffer;
      options.onPresentation?.(presentation);
      stimulus.max = String(first.total_rows);
      columns = Math.ceil(Math.sqrt(buffer.length)); rows = Math.ceil(buffer.length / columns);
      cell = Math.max(1, Math.min(16, Math.floor(Math.min(Math.max(200, container.clientWidth - 8) / columns, 420 / rows))));
      canvas.width = columns; canvas.height = rows; selected = 0; layout();
      setExtent(first.extent ?? 1);
      const slice = Object.entries(first.slice).filter(([dim]) => dim !== 'time_bin').map(([dim, value]) => `${dim}=${value}`).join(', ');
      status.textContent = `All ${buffer.length.toLocaleString()} recorded neurons · stimulus ${presentation + 1} of ${first.total_rows}: ${page.stimulus}${first.time ? ` · ${timeLabel(first.time)}` : ''}${slice ? ` · fixed slice: ${slice}` : ''}`;
      detail.textContent = 'Click a neuron, or focus the heatmap and use arrow keys to inspect values.';
    } catch (error) {
      if (current !== sequence || controller.signal.aborted) return;
      status.textContent = String(error); values = null;
      canvas.getContext('2d')!.clearRect(0, 0, canvas.width, canvas.height);
    }
  };
  stimulus.onchange = () => void load();
  zoom.oninput = layout;
  canvas.onclick = event => {
    const size = cell * Number(zoom.value) / 100;
    const column = Math.floor(event.offsetX / size), row = Math.floor(event.offsetY / size);
    const index = row * columns + column;
    if (values && index < values.length) void inspect(index);
  };
  canvas.onkeydown = event => {
    const offsets: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns, Home: -selected, End: (values?.length ?? 1) - 1 - selected };
    if (!(event.key in offsets)) return;
    event.preventDefault(); void inspect(selected + offsets[event.key], true);
  };
  queueMicrotask(() => { if (container.isConnected) void load(); });
  draw(extent);
  return container;
}
