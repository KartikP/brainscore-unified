# Temporal neural views

The lab currently shows index zero on every axis except presentation and
neuroid. Add a time-bin selector to snapshot and stimulus comparison views.
Add a neuron-by-time heatmap and traces for up to eight explicit neurons.

Use saved arrays, never model calls. Return native time-bin bounds from plain
coordinates or recorded MultiIndex levels. When physical bounds are absent,
label the horizontal axis as bin index. Never invent milliseconds, resample,
average, or join different channel clocks. Keep missing values visible.

Preserve the existing paged snapshot endpoint and add an optional time_bin
parameter with default zero. Add a bounded temporal endpoint that reads one
presentation, explicit neurons and a time window through NumPy memory maps.
Reject oversized requests before data materialization. Expose page controls
and ranges; no silent neuron or time sampling. Preserve other fixed axes in
the displayed metadata.

Keep TypeScript and canvas for browser plots. Use native controls, accessible
labels, keyboard inspection, and the existing neutral visual style. Temporal
heatmap colors use the existing shared scale. Traces use physical intervals
where available, with breaks at missing values and interval gaps.

Alternatives: a slider alone leaves temporal patterns hard to compare; a full
array download creates unnecessary memory use. Bounded saved-data windows
support both detailed inspection and large recordings.

Validate exact saved values, dimension order, channel selection, native time
metadata, missing values, request limits and HTTP errors remotely. Build and
test the frontend remotely, then refresh the packaged assets locally.
