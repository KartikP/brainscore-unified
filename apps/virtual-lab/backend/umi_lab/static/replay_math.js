import { dataURL } from './temporal_math.js';
export const MAX_REPLAY_VALUES = 4_000_000;
export function checkReplayBudget(channels) {
    let values = 0;
    for (const channel of channels) {
        if (!Number.isSafeInteger(channel.units) || channel.units < 1)
            throw new Error('Invalid recorded neuron count');
        values += channel.units * Math.max(1, channel.time_bins.length);
    }
    if (!Number.isSafeInteger(values) || values > MAX_REPLAY_VALUES)
        throw new Error('This selection exceeds the replay budget of 4,000,000 values. Use the paged temporal views in Results. No neurons or bins have been omitted.');
}
// Negative values mean no interval (-1) or ambiguous overlapping intervals (-2).
export function binAtTime(bins, milliseconds) {
    if (!Number.isFinite(milliseconds))
        return -1;
    let match = -1;
    bins.forEach((bin, index) => {
        if (bin.start_ms === null || bin.end_ms === null || bin.end_ms <= bin.start_ms)
            return;
        if (milliseconds >= bin.start_ms && (milliseconds < bin.end_ms || index === bins.length - 1 && milliseconds === bin.end_ms)) {
            match = match === -1 ? index : -2;
        }
    });
    return match;
}
export async function loadReplayChannel(source, channel, presentation, get, signal, progress = () => { }) {
    checkReplayBudget([channel]);
    const times = Math.max(1, channel.time_bins.length), units = channel.units;
    const values = new Float64Array(times * units);
    let loaded = 0;
    const assign = (value, index) => {
        if (value !== null && (typeof value !== 'number' || !Number.isFinite(value)))
            throw new Error('Invalid saved neural value');
        values[index] = value === null ? NaN : value;
    };
    for (let time = 0; time < times; time += 128) {
        const count = Math.min(128, times - time), pageSize = Math.min(2048, Math.floor(65536 / count));
        for (let offset = 0; offset < units; offset += pageSize) {
            if (signal.aborted)
                throw new Error('Replay load cancelled');
            const limit = Math.min(pageSize, units - offset);
            const temporal = channel.time_bins.length > 0;
            const page = await get(dataURL(source, { presentation, offset, limit, labels: 0,
                ...(temporal ? { time_offset: time, time_limit: count } : {}) }, temporal));
            if (signal.aborted)
                throw new Error('Replay load cancelled');
            if (page.total_columns !== units || page.presentation !== presentation || page.offset !== offset)
                throw new Error('Inconsistent saved replay page');
            if (temporal) {
                if (page.total_time_bins !== times || page.time_offset !== time || page.values.length !== count ||
                    page.neurons.length !== limit || page.neurons.some((neuron, index) => neuron !== offset + index))
                    throw new Error('Inconsistent saved replay time window');
                page.values.forEach((row, step) => {
                    if (row.length !== limit)
                        throw new Error('Incomplete saved replay neurons');
                    row.forEach((value, neuron) => assign(value, (time + step) * units + offset + neuron));
                });
            }
            else {
                if (page.values.length !== limit)
                    throw new Error('Incomplete saved replay neurons');
                page.values.forEach((value, neuron) => assign(value, offset + neuron));
            }
            loaded += count * limit;
            progress(loaded);
        }
    }
    return values;
}
