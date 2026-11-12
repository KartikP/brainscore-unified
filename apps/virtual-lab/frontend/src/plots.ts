import { HeatmapScale } from './full_heatmap.js';
import { neuralViews } from './neural_comparison.js';
export { HeatmapScale };

export interface ActivationPreview {
  values: (number | null)[][];
  row_labels: string[];
  column_labels: string[];
  total_rows: number;
  total_columns: number;
  row_axis: string;
  column_axis: string;
  slice: Record<string, number | string>;
  condition?: string;
  target?: string;
}
interface ChoiceRow { stimulus: string; stimulus_text?: string | null; scores: (number | null)[]; selection_status?: string }
interface TraceRow { raw_answer?: string; valid?: boolean }
const names: Record<string, string> = { baseline: 'Baseline', intervention: 'Intervention', ablation: 'Ablation', random_control: 'Random control', restored: 'Restored' };
const element = <K extends keyof HTMLElementTagNameMap>(tag: K, className = '', text = '') => {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
};
const number = (value: number) => Number.isInteger(value) ? String(value) : value.toPrecision(4);

export function barDomain(values: (number | null)[]): [number, number] {
  const finite = values.filter((value): value is number => value !== null && Number.isFinite(value));
  const low = Math.min(0, ...finite), high = Math.max(0, ...finite);
  return low === high ? [0, 1] : [low, high];
}

export function answerCounts(rows: TraceRow[]): { label: string; value: number }[] {
  const counts = new Map<string, number>();
  for (const row of rows) {
    if (typeof row.raw_answer === 'string') counts.set(row.raw_answer, (counts.get(row.raw_answer) ?? 0) + 1);
  }
  return [...counts].map(([label, value]) => ({ label, value }));
}

export function activationExtent(previews: ActivationPreview[]): number {
  let extent = 0;
  for (const preview of previews) for (const row of preview.values) for (const value of row) {
    if (value !== null && Number.isFinite(value)) extent = Math.max(extent, Math.abs(value));
  }
  return extent || 1;
}

function bars(items: { label: string; value: number | null }[], percent = false): HTMLElement {
  const chart = element('div', 'bar-chart');
  const [low, high] = percent ? [0, 1] : barDomain(items.map(item => item.value));
  const position = (value: number) => 100 * (value - low) / (high - low);
  for (const item of items) {
    const row = element('div', 'bar-item');
    const label = element('span', 'bar-label', item.label || '(empty response)'); label.title = item.label;
    const track = element('div', 'bar-track'); track.setAttribute('aria-hidden', 'true');
    const zero = element('span', 'bar-zero'); zero.style.left = `${position(0)}%`; track.append(zero);
    const finite = item.value !== null && Number.isFinite(item.value);
    if (finite) {
      const fill = element('span', 'bar-fill');
      fill.style.left = `${Math.min(position(0), position(item.value!))}%`;
      fill.style.width = `${Math.abs(position(item.value!) - position(0))}%`;
      if (item.value! < 0) fill.classList.add('negative');
      track.append(fill);
    }
    const value = element('span', 'bar-value', finite ? percent ? `${(item.value! * 100).toFixed(0)}%` : number(item.value!) : 'Missing');
    row.append(label, track, value); chart.append(row);
  }
  chart.append(element('p', 'plot-axis', `${percent ? '0%' : number(low)} — ${percent ? '100%' : number(high)}`));
  return chart;
}

function choices(response: Record<string, unknown>): HTMLElement {
  const rows = response.responses as ChoiceRow[];
  const labels = response.choice_labels as string[] | undefined;
  const container = element('div');
  const label = element('label', 'plot-stimulus', 'Stimulus');
  const select = document.createElement('select'); select.setAttribute('aria-label', 'Stimulus to visualize');
  rows.forEach((row, index) => select.append(new Option(`${row.stimulus}: ${row.stimulus_text ?? ''}`.slice(0, 160), String(index))));
  label.append(select);
  const plot = element('div');
  const render = () => {
    const row = rows[Number(select.value)];
    plot.replaceChildren(bars(row.scores.map((value, index) => ({ label: labels?.[index] ?? `Choice ${index + 1}`, value }))));
    plot.append(element('p', 'subtle', row.selection_status === 'tie' ? 'Equal maxima: no preference.' : row.selection_status === 'invalid' ? 'Contains nonfinite values; no preference reported.' : 'Raw choice values. These are not accuracy scores.'));
  };
  select.onchange = render; render(); container.append(label, plot);
  const shape = response.shape as number[] | undefined;
  const dims = response.dims as string[] | undefined;
  const totalChoices = shape?.[dims?.indexOf('choice') ?? -1] ?? rows[0].scores.length;
  container.append(element('p', 'subtle', `Showing ${rows.length} recorded stimulus rows and the first ${rows[0].scores.length} of ${totalChoices} choices. Complete data are in the artifacts.`));
  return container;
}

function frequencies(response: Record<string, unknown>): HTMLElement {
  const rows = response.preview as TraceRow[];
  const container = element('div');
  container.append(bars(answerCounts(rows)));
  const invalid = rows.filter(row => row.valid === false).length;
  container.append(element('p', 'subtle', `Exact response counts for ${rows.length} previewed events of ${response.count ?? rows.length}. Case and punctuation are preserved; these are not accuracy scores.${invalid ? ` ${invalid} events have parse errors; their raw answers remain included.` : ''}`));
  return container;
}

function heatmap(preview: ActivationPreview, extent: number): HTMLElement {
  const container = element('div', 'activation-plot');
  const scroll = element('div', 'heatmap-scroll');
  const table = element('table', 'activation-grid');
  const caption = document.createElement('caption'); caption.textContent = `${preview.row_axis} × ${preview.column_axis}: raw activations`; table.append(caption);
  const head = document.createElement('tr');
  head.append(element('th', 'heatmap-axis', preview.row_axis));
  for (const label of preview.column_labels) { const cell = element('th', '', label); cell.scope = 'col'; cell.title = label; head.append(cell); }
  table.append(head);
  const selected = element('p', 'plot-selection', 'Select a cell to inspect its value. Use arrow keys to move between cells.');
  selected.setAttribute('aria-live', 'polite');
  const cells: HTMLButtonElement[] = [];
  const columns = preview.column_labels.length;
  const choose = (index: number, focus = false) => {
    cells.forEach((cell, i) => { cell.tabIndex = i === index ? 0 : -1; cell.setAttribute('aria-pressed', String(i === index)); });
    const row = Math.floor(index / columns), column = index % columns;
    const value = preview.values[row][column];
    selected.textContent = `${preview.row_axis} ${preview.row_labels[row]} / ${preview.column_axis} ${preview.column_labels[column]}: ${value === null ? 'nonfinite / missing' : number(value)}`;
    if (focus) cells[index].focus();
  };
  preview.values.forEach((values, row) => {
    const tr = document.createElement('tr');
    const label = element('th', 'heatmap-row', preview.row_labels[row]); label.scope = 'row'; label.title = preview.row_labels[row]; tr.append(label);
    values.forEach((value, column) => {
      const cell = document.createElement('td'); const button = element('button', 'activation-cell'); button.type = 'button';
      const index = cells.length; button.tabIndex = index === 0 ? 0 : -1;
      button.setAttribute('aria-pressed', 'false');
      const description = `${preview.row_axis} ${preview.row_labels[row]}, ${preview.column_axis} ${preview.column_labels[column]}: ${value === null ? 'nonfinite / missing' : number(value)}`;
      button.title = description; button.setAttribute('aria-label', description);
      if (value === null || !Number.isFinite(value)) button.classList.add('missing');
      else {
        const color = value < 0 ? [72, 111, 157] : [181, 119, 58];
        const intensity = Math.min(1, Math.abs(value) / extent);
        button.style.background = `rgb(${color.map(component => Math.round(250 + (component - 250) * intensity)).join(',')})`;
      }
      button.onclick = () => choose(index);
      button.onkeydown = event => {
        const delta: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns };
        if (!(event.key in delta)) return;
        event.preventDefault(); choose(Math.max(0, Math.min(cells.length - 1, index + delta[event.key])), true);
      };
      cells.push(button); cell.append(button); tr.append(cell);
    });
    table.append(tr);
  });
  scroll.append(table); container.append(scroll);
  const legend = element('div', 'activation-legend');
  legend.append(element('span', '', `−${number(extent)}`), element('span', 'activation-ramp'), element('span', '', `+${number(extent)}`), element('span', '', '0 = neutral · gray = missing'));
  container.append(legend, selected);
  const slice = Object.entries(preview.slice).map(([dim, index]) => `${dim}=${index}`).join(', ');
  container.append(element('p', 'subtle', `First ${preview.values.length} of ${preview.total_rows} ${preview.row_axis.toLowerCase()}s × ${columns} of ${preview.total_columns} ${preview.column_axis.toLowerCase()}s.${slice ? ` Fixed slice: ${slice}.` : ''} Raw values; shared color scale across displayed conditions. No averaging or normalization.`));
  return container;
}

function activationData(preview: ActivationPreview): HTMLElement {
  const scroll = element('div', 'heatmap-scroll');
  const table = element('table', 'response-table activation-data');
  const header = document.createElement('tr'); header.append(element('th', '', preview.row_axis));
  for (const label of preview.column_labels) header.append(element('th', '', label));
  table.append(header);
  preview.values.forEach((row, index) => {
    const tr = document.createElement('tr');
    const label = element('th', '', preview.row_labels[index]); label.scope = 'row'; tr.append(label);
    for (const value of row) tr.append(element('td', '', value === null ? 'Missing' : number(value)));
    table.append(tr);
  });
  scroll.append(table, element('p', 'subtle', 'The same bounded preview as the heatmap. Full precision and complete arrays are saved in the artifacts.'));
  return scroll;
}

function viewControls(card: HTMLElement, plot: HTMLElement, data: HTMLElement[], title: string): void {
  const heading = element('div', 'plot-heading');
  heading.append(element('h4', '', title));
  const controls = element('div', 'plot-controls'); controls.setAttribute('role', 'group'); controls.setAttribute('aria-label', `${title} view`);
  const plotButton = element('button', '', 'Plot'), dataButton = element('button', '', 'Data');
  plotButton.type = dataButton.type = 'button';
  const show = (visible: boolean) => {
    plot.hidden = !visible; data.forEach(item => { item.hidden = visible; });
    plotButton.setAttribute('aria-pressed', String(visible)); dataButton.setAttribute('aria-pressed', String(!visible));
  };
  plotButton.onclick = () => show(true); dataButton.onclick = () => show(false);
  controls.append(plotButton, dataButton); heading.append(controls); card.append(heading, plot, ...data); show(true);
}

export function addResponsePlot(card: HTMLElement, response: Record<string, unknown>, extent: number, source?: string, scale = new HeatmapScale()): void {
  const data = [...card.children].filter((node): node is HTMLElement => node instanceof HTMLElement && node.tagName !== 'H3' && !node.classList.contains('response-meta'));
  const activation = response.activation_preview as ActivationPreview | undefined;
  if (activation?.values.length && activation.column_labels.length) {
    data.forEach(node => node.remove());
    viewControls(card, source ? neuralViews(source, scale, activation.row_labels) : heatmap(activation, extent), [activationData(activation)], 'Activation probe');
  }
  else if ((response.responses as ChoiceRow[] | undefined)?.length) viewControls(card, choices(response), data, 'Behavior');
  else if (Array.isArray(response.scalar_rows)) {
    const rows = response.scalar_rows as { stimulus: string; value: number | null }[];
    const plot = element('div');
    plot.append(bars(rows.map(row => ({ label: row.stimulus, value: row.value }))));
    plot.append(element('p', 'subtle', 'Raw recorded values for the previewed stimuli. Missing or invalid answers are not replaced.'));
    viewControls(card, plot, data, 'Behavior');
  }
  else if (Array.isArray(response.preview) && response.preview.some(row => typeof row?.raw_answer === 'string')) viewControls(card, frequencies(response), data, 'Response frequency');
}

export function addCalibrationPlots(report: Record<string, unknown>, container: HTMLElement): void {
  const conditions = report.conditions as { condition: string; accuracy: number }[];
  const behavior = element('div', 'response-card');
  behavior.append(element('h3', '', 'Calibration behavior'));
  behavior.append(bars(conditions.map(row => ({ label: names[row.condition] ?? row.condition, value: row.accuracy })), true));
  container.append(behavior);
  const previews = report.activation_previews as ActivationPreview[] | undefined;
  const extent = activationExtent(previews ?? []);
  for (const preview of previews ?? []) {
    const card = element('div', 'response-card');
    card.append(element('h3', '', `${names[preview.condition ?? ''] ?? preview.condition} / ${preview.target}`));
    card.append(heatmap(preview, extent)); container.append(card);
  }
}
