import { dataURL, neuronIndices, physicalIntervals, timeLabel, traceSegments } from './temporal_math.js';
const node = (tag, text = '', className = '') => {
    const element = document.createElement(tag);
    element.textContent = text;
    element.className = className;
    return element;
};
const format = (value) => Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : 'missing';
const traceColors = ['#486f9d', '#a16732', '#527d60', '#885e8b', '#536771', '#a14e50', '#766b3b', '#596ba0'];
export function temporalView(source, scale, stimulusLabels, traces) {
    const panel = node('div', '', 'temporal-view'), controls = node('div', '', 'temporal-controls');
    function numeric(label, value, max) {
        const wrapper = node('label', label), input = node('input');
        input.type = 'number';
        input.min = '1';
        input.max = String(max);
        input.step = '1';
        input.value = String(value);
        input.setAttribute('aria-label', label);
        wrapper.append(input);
        controls.append(wrapper);
        return input;
    }
    const stimulus = numeric('Stimulus', 1, 1), firstTime = numeric('First time bin', 1, 1);
    const timeCount = numeric('Time bins per page', traces ? 512 : 128, 1024);
    const firstNeuron = traces ? undefined : numeric('First neuron (1-based)', 1, 1);
    const neuronCount = traces ? undefined : numeric('Neurons per page', 256, 2048);
    const neuronsLabel = node('label', 'Neuron indices (0-based)'), neurons = node('input');
    neurons.type = 'text';
    neurons.value = '0';
    neurons.setAttribute('aria-label', 'Neuron indices (0-based)');
    neuronsLabel.append(neurons);
    if (traces)
        controls.append(neuronsLabel);
    const previous = node('button', 'Previous time page'), next = node('button', 'Next time page');
    previous.type = next.type = 'button';
    controls.append(previous, next);
    const status = node('p', 'Read a saved temporal window.', 'subtle');
    status.setAttribute('role', 'status');
    const plot = node('div'), detail = node('p', '', 'plot-selection');
    detail.setAttribute('aria-live', 'polite');
    const note = node('p', traces
        ? 'Each trace uses the selected neuron’s raw values. Native intervals appear as constant segments; missing values and time gaps break the trace. Without physical bounds, the axis uses bin indices.'
        : 'Rows are neurons; columns follow the recorded time-bin order. Each cell is one raw value. Column widths are equal; the detail shows actual time bounds. Gray means missing. Page controls expose every recorded neuron and time bin.', 'subtle');
    panel.append(controls, status, plot, detail, note);
    let page, controller, sequence = 0, sharedExtent = 1;
    let selectedTime = 0, selectedNeuron = 0;
    let selection, cellWidth = 1;
    const setExtent = scale.register(panel, high => { sharedExtent = high; if (page && !traces)
        drawHeatmap(); });
    const description = (time, neuron) => {
        if (!page)
            return '';
        return `${timeLabel(page.time_bins[time])} / neuron ${page.neurons[neuron]} (${page.labels[neuron]}): ${page.values[time][neuron] === null ? 'missing' : format(page.values[time][neuron])}`;
    };
    const inspect = (time, neuron) => {
        if (!page)
            return;
        selectedTime = Math.max(0, Math.min(page.time_bins.length - 1, time));
        selectedNeuron = Math.max(0, Math.min(page.neurons.length - 1, neuron));
        detail.textContent = description(selectedTime, selectedNeuron);
        if (selection) {
            selection.style.left = `${selectedTime * cellWidth}px`;
            selection.style.top = `${selectedNeuron * 3}px`;
        }
    };
    function drawHeatmap() {
        if (!page)
            return;
        const width = page.time_bins.length, height = page.neurons.length, canvas = node('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.tabIndex = 0;
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', `${height} neurons across ${width} recorded time bins. Columns are time bins; rows are neurons. Use arrow keys to inspect raw values.`);
        const context = canvas.getContext('2d'), pixels = context.createImageData(width, height);
        for (let time = 0; time < width; time++)
            for (let neuron = 0; neuron < height; neuron++) {
                const value = page.values[time][neuron], missing = value === null, offset = (neuron * width + time) * 4;
                const target = (value ?? 0) < 0 ? [72, 111, 157] : [181, 119, 58];
                const strength = missing ? 0 : Math.min(1, Math.abs(value) / sharedExtent);
                for (let channel = 0; channel < 3; channel++)
                    pixels.data[offset + channel] = missing ? 220 : Math.round(250 + (target[channel] - 250) * strength);
                pixels.data[offset + 3] = 255;
            }
        context.putImageData(pixels, 0, 0);
        cellWidth = Math.max(2, Math.min(12, Math.floor(720 / width)));
        const cellHeight = 3;
        canvas.style.width = `${width * cellWidth}px`;
        canvas.style.height = `${height * cellHeight}px`;
        const scroll = node('div', '', 'full-heatmap-scroll temporal-heatmap'), surface = node('div', '', 'temporal-surface');
        surface.style.width = canvas.style.width;
        surface.style.height = canvas.style.height;
        selection = node('div', '', 'heatmap-selection');
        selection.style.width = `${cellWidth}px`;
        selection.style.height = `${cellHeight}px`;
        surface.append(canvas, selection);
        scroll.append(surface);
        const legend = node('div', '', 'activation-legend');
        legend.append(node('span', `-${format(sharedExtent)}`), node('span', '', 'activation-ramp'), node('span', `+${format(sharedExtent)}`), node('span', '0 = neutral; gray = missing'));
        plot.replaceChildren(node('p', 'Horizontal: time bins. Vertical: neurons in recorded order.', 'subtle'), scroll, legend);
        canvas.onclick = event => inspect(Math.floor(event.offsetX / cellWidth), Math.floor(event.offsetY / cellHeight));
        canvas.onkeydown = event => {
            const offsets = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
                Home: [-selectedTime, -selectedNeuron], End: [width - 1 - selectedTime, height - 1 - selectedNeuron] };
            if (event.key in offsets) {
                event.preventDefault();
                const [time, neuron] = offsets[event.key];
                inspect(selectedTime + time, selectedNeuron + neuron);
                selection?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
            }
        };
        inspect(selectedTime, selectedNeuron);
    }
    function drawTraces() {
        if (!page)
            return;
        const physical = physicalIntervals(page.time_bins), bins = page.time_bins;
        let minX = Infinity, maxX = -Infinity, minY = 0, maxY = 0;
        bins.forEach(bin => {
            minX = Math.min(minX, physical ? bin.start_ms : bin.index + 1);
            maxX = Math.max(maxX, physical ? bin.end_ms : bin.index + 1);
        });
        page.values.forEach(row => row.forEach(value => { if (value !== null) {
            minY = Math.min(minY, value);
            maxY = Math.max(maxY, value);
        } }));
        if (minY === maxY) {
            minY -= 1;
            maxY += 1;
        }
        const spanX = maxX - minX || 1, width = 840, height = 320, left = 70, right = 24, top = 20, bottom = 50;
        const x = (value) => left + (value - minX) / spanX * (width - left - right);
        const y = (value) => height - bottom - (value - minY) / (maxY - minY) * (height - top - bottom);
        const canvas = node('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.className = 'neural-traces';
        canvas.tabIndex = 0;
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', `Raw neural response traces for ${page.neurons.length} neurons. Horizontal axis: ${physical ? 'milliseconds' : 'time-bin index'}. Use left and right arrow keys to inspect bins; use up and down to select a trace.`);
        const context = canvas.getContext('2d');
        context.fillStyle = '#fff';
        context.fillRect(0, 0, width, height);
        context.font = '12px system-ui';
        context.fillStyle = '#525b65';
        context.strokeStyle = '#dce0e4';
        context.lineWidth = 1;
        for (let tick = 0; tick <= 4; tick++) {
            const value = minY + (maxY - minY) * tick / 4;
            context.beginPath();
            context.moveTo(left, y(value));
            context.lineTo(width - right, y(value));
            context.stroke();
            context.textAlign = 'right';
            context.fillText(format(value), left - 8, y(value) + 4);
            const time = minX + spanX * tick / 4;
            context.textAlign = 'center';
            context.fillText(format(time), x(time), height - bottom + 20);
        }
        context.fillText(physical ? 'Time (ms)' : 'Time-bin index (1-based)', (left + width - right) / 2, height - 7);
        context.save();
        context.translate(15, height / 2);
        context.rotate(-Math.PI / 2);
        context.fillText('Raw activation', 0, 0);
        context.restore();
        const legend = node('div', '', 'trace-legend');
        page.neurons.forEach((neuron, column) => {
            context.strokeStyle = context.fillStyle = traceColors[column];
            context.lineWidth = 1.6;
            for (const segment of traceSegments(bins, page.values.map(row => row[column]), physical)) {
                context.beginPath();
                segment.forEach((point, index) => index ? context.lineTo(x(point.x), y(point.y)) : context.moveTo(x(point.x), y(point.y)));
                context.stroke();
                if (segment.length === 1) {
                    context.beginPath();
                    context.arc(x(segment[0].x), y(segment[0].y), 2, 0, 2 * Math.PI);
                    context.fill();
                }
            }
            const entry = node('span', `Neuron ${neuron} (${page.labels[column]})`);
            entry.style.color = traceColors[column];
            legend.append(entry);
        });
        const scroll = node('div', '', 'trace-scroll');
        scroll.append(canvas);
        plot.replaceChildren(scroll, legend);
        canvas.onclick = event => {
            const px = event.offsetX * width / canvas.clientWidth, target = minX + (px - left) / (width - left - right) * spanX;
            let nearest = 0, distance = Infinity;
            bins.forEach((bin, index) => {
                const middle = physical ? (bin.start_ms + bin.end_ms) / 2 : bin.index + 1;
                if (Math.abs(middle - target) < distance) {
                    nearest = index;
                    distance = Math.abs(middle - target);
                }
            });
            inspect(nearest, selectedNeuron);
        };
        canvas.onkeydown = event => {
            const offsets = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
            if (event.key in offsets) {
                event.preventDefault();
                const [time, neuron] = offsets[event.key];
                inspect(selectedTime + time, selectedNeuron + neuron);
            }
        };
    }
    const load = async () => {
        controller?.abort();
        controller = new AbortController();
        const signal = controller.signal, current = ++sequence;
        page = undefined;
        plot.replaceChildren();
        detail.textContent = '';
        status.textContent = 'Read saved temporal values...';
        try {
            const integer = (input, max) => {
                const value = Number(input.value);
                if (!Number.isSafeInteger(value) || value < 1 || value > max)
                    throw new Error(`Choose ${input.getAttribute('aria-label')} from 1 to ${max}.`);
                return value;
            };
            const settings = { presentation: integer(stimulus, Number(stimulus.max)) - 1,
                time_offset: integer(firstTime, Number(firstTime.max)) - 1, time_limit: integer(timeCount, 1024) };
            if (traces)
                settings.neurons = neuronIndices(neurons.value).join(',');
            else {
                settings.offset = integer(firstNeuron, Number(firstNeuron.max)) - 1;
                settings.limit = integer(neuronCount, 2048);
            }
            const response = await fetch(dataURL(source, settings, true), { signal });
            const result = await response.json();
            if (!response.ok)
                throw new Error(result.error ?? 'Saved temporal values unavailable');
            if (current !== sequence || !panel.isConnected)
                return;
            page = result;
            stimulus.max = String(page.total_rows);
            firstTime.max = String(page.total_time_bins);
            if (firstNeuron)
                firstNeuron.max = String(page.total_columns);
            previous.disabled = page.time_offset === 0;
            next.disabled = page.time_offset + page.time_bins.length >= page.total_time_bins;
            const fixed = Object.entries(page.slice).map(([dim, value]) => `${dim}=${value}`).join(', ');
            const units = traces ? `Neurons ${page.neurons.join(', ')} of ${page.total_columns}` : `Neurons ${page.neurons[0] + 1} to ${page.neurons.at(-1) + 1} of ${page.total_columns}`;
            status.textContent = `Stimulus ${page.presentation + 1} of ${page.total_rows}: ${stimulusLabels[page.presentation] ?? page.stimulus}. ${units}. Time bins ${page.time_offset + 1} to ${page.time_offset + page.time_bins.length} of ${page.total_time_bins}.${fixed ? ` Fixed slice: ${fixed}.` : ''} Saved data only.`;
            selectedTime = selectedNeuron = 0;
            if (traces)
                drawTraces();
            else
                setExtent(page.extent);
            inspect(0, 0);
        }
        catch (error) {
            if (current === sequence && !signal.aborted)
                status.textContent = String(error);
        }
    };
    for (const input of [stimulus, firstTime, timeCount, firstNeuron, neuronCount, ...(traces ? [neurons] : [])])
        if (input)
            input.onchange = () => void load();
    previous.onclick = () => { firstTime.value = String(Math.max(1, Number(firstTime.value) - Number(timeCount.value))); void load(); };
    next.onclick = () => { if (page)
        firstTime.value = String(page.time_offset + page.time_bins.length + 1); void load(); };
    queueMicrotask(() => { if (panel.isConnected)
        void load(); });
    return panel;
}
