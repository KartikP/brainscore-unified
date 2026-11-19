// Descriptive comparisons within one saved channel, condition and repetition.
export function difference(a, b) {
    if (a.length !== b.length)
        throw new Error('Recordings must have matching units');
    return Float64Array.from(a, (value, index) => Number.isFinite(value) && Number.isFinite(b[index]) ? value - b[index] : NaN);
}
export function largestChanges(values, limit = 20) {
    return Array.from(values.keys()).filter(index => Number.isFinite(values[index]))
        .sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]) || a - b).slice(0, limit);
}
export function variableUnits(rows, limit = 32) {
    const width = rows[0]?.length ?? 0;
    if (rows.some(row => row.length !== width))
        throw new Error('Recordings must have matching units');
    const means = new Float64Array(width), ranges = new Float64Array(width);
    for (let unit = 0; unit < width; unit++) {
        let low = Infinity, high = -Infinity, sum = 0, count = 0;
        for (const row of rows)
            if (Number.isFinite(row[unit])) {
                low = Math.min(low, row[unit]);
                high = Math.max(high, row[unit]);
                sum += row[unit];
                count++;
            }
        means[unit] = count ? sum / count : NaN;
        ranges[unit] = count >= 2 ? high - low : NaN;
    }
    return { indices: largestChanges(ranges, limit), means };
}
export function distance(a, b, metric) {
    if (a.length !== b.length)
        throw new Error('Recordings must have matching units');
    let count = 0, squared = 0, meanA = 0, meanB = 0, ssA = 0, ssB = 0, cross = 0;
    for (let index = 0; index < a.length; index++) {
        const x = a[index], y = b[index];
        if (!Number.isFinite(x) || !Number.isFinite(y))
            continue;
        count++;
        squared += (x - y) ** 2;
        // Stable online covariance; persistent feature offsets may still dominate correlation.
        const dx = x - meanA, dy = y - meanB;
        meanA += dx / count;
        meanB += dy / count;
        ssA += dx * (x - meanA);
        ssB += dy * (y - meanB);
        cross += dx * (y - meanB);
    }
    if (!count)
        return { value: null, count };
    const value = metric === 'rms' ? Math.sqrt(squared / count) : count < 2 || ssA <= 0 || ssB <= 0
        ? null : 1 - Math.max(-1, Math.min(1, cross / Math.sqrt(ssA * ssB)));
    return { value, count };
}
export const MAX_COMPARISON_CELLS = 1_000_000;
export function checkComparisonSize(rows, columns) {
    if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(columns) || rows < 2 || columns < 1)
        throw new Error('Comparisons need at least two recorded stimuli and one unit.');
    if (rows > 64 || columns > 65536 || rows * columns > MAX_COMPARISON_CELLS)
        throw new Error('This recording exceeds the interactive comparison budget (64 stimuli, 65,536 units, 1,000,000 values). Use the exported arrays; no stimuli or units have been silently sampled.');
}
export async function loadComparisons(get, signal) {
    const first = await get(0, 0);
    checkComparisonSize(first.total_rows, first.total_columns);
    const rows = [], labels = [];
    for (let presentation = 0; presentation < first.total_rows; presentation++) {
        const row = new Float64Array(first.total_columns);
        let offset = 0;
        while (offset < row.length) {
            if (signal.aborted)
                throw new Error('Comparison loading cancelled');
            const page = presentation === 0 && offset === 0 ? first : await get(presentation, offset);
            if (page.total_rows !== first.total_rows || page.total_columns !== first.total_columns ||
                page.presentation !== presentation || page.offset !== offset || page.values.length < 1 ||
                page.values.length > 2048 || offset + page.values.length > row.length ||
                JSON.stringify(page.slice) !== JSON.stringify(first.slice))
                throw new Error('Inconsistent saved activation pages');
            page.values.forEach((value, index) => { row[offset + index] = value ?? NaN; });
            labels[presentation] = page.stimulus;
            offset += page.values.length;
        }
        rows.push(row);
    }
    return { rows, labels, slice: first.slice };
}
