export function dataURL(source, settings, temporal = false) {
    const url = new URL(source, 'http://lab.local');
    if (temporal)
        url.pathname = url.pathname.replace(/\/activation$/, '/temporal');
    for (const [name, value] of Object.entries(settings))
        url.searchParams.set(name, String(value));
    return url.pathname + url.search;
}
const format = (value) => Number(value.toPrecision(6)).toString();
export function timeLabel(bin) {
    if (bin.start_ms !== null && bin.end_ms !== null)
        return `Bin ${bin.index + 1}: ${format(bin.start_ms)} to ${format(bin.end_ms)} ms`;
    if (bin.center_ms !== null)
        return `Bin ${bin.index + 1}: center ${format(bin.center_ms)} ms`;
    return `Bin ${bin.index + 1}: physical time unavailable`;
}
export function physicalIntervals(bins) {
    return bins.length > 0 && bins.every(bin => bin.start_ms !== null && bin.end_ms !== null &&
        Number.isFinite(bin.start_ms) && Number.isFinite(bin.end_ms) && bin.end_ms >= bin.start_ms);
}
export function traceSegments(bins, values, physical = physicalIntervals(bins)) {
    if (bins.length !== values.length)
        throw new Error('Time bins and trace values must have equal lengths');
    const result = [];
    let segment, previousEnd;
    bins.forEach((bin, index) => {
        const value = values[index];
        if (value === null || !Number.isFinite(value)) {
            segment = undefined;
            previousEnd = undefined;
            return;
        }
        const start = physical ? bin.start_ms : bin.index + 1;
        const end = physical ? bin.end_ms : bin.index + 1;
        const contiguous = physical ? previousEnd === start : previousEnd === start - 1;
        if (!segment || !contiguous) {
            segment = [];
            result.push(segment);
        }
        segment.push({ x: start, y: value });
        if (end !== start)
            segment.push({ x: end, y: value });
        previousEnd = end;
    });
    return result;
}
export function neuronIndices(text) {
    const parts = text.split(',').map(part => part.trim());
    if (parts.length < 1 || parts.length > 8 || parts.some(part => !/^\d+$/.test(part)))
        throw new Error('Enter 1 to 8 neuron indices, separated by commas. Indices start at 0.');
    const values = parts.map(Number);
    if (values.some(value => !Number.isSafeInteger(value)) || new Set(values).size !== values.length)
        throw new Error('Neuron indices must be distinct nonnegative integers.');
    return values;
}
