# Saved neural replay

Add Results and Replay tabs to experiment review. Replay selects a saved
condition, repetition and stimulus. It displays the original saved media and
every recorded neural region. No subject, model weights, inference or fMRI
mapping is part of this feature.

Resolve stimuli by saved IDs, not presentation positions. Read each region's
native time bounds. The media clock selects a bin independently in each region.
Display gaps, overlapping bins and missing clocks explicitly. Static outputs
remain static. Never interpolate bins or imply that transformer outputs are
causal online responses. Explain that timestamps locate recorded output bins.

Load bounded pages into a selected-stimulus replay buffer before Play. Cap the
combined neural buffer at four million values. Include every neuron and reject
oversized selections with a route to the Results temporal viewer. Fix each
region's color range across the inspected clip; do not compare magnitudes across
regions. Cache no other stimuli. Pause on tab, condition or stimulus changes.

Serve only media files declared in saved stimulus rows and copied into that
run's inputs directory. Reject path traversal, escaping symlinks, active image
formats and invalid ranges. Support HTTP byte ranges for media seek. Display
text/images directly; use native video/audio playback with an explicit seek
control. For video plus separate audio, use the audible stream's clock and let
the silent video follow it with bounded corrections. Never repeatedly seek audio
during steady playback. Without physical time, show static responses and an honest
alignment message instead of inventing a duration or clock.

The alternative of a separate player for each region would obscure time
comparisons. Streaming values from a live model would add inference cost and
change the meaning of replay. This design uses only saved artifacts.

Validate ID alignment, selection of repetitions and conditions, independent
clocks, gaps, array budgets, media access and ranges remotely. Use synthetic
media for browser tests of Play, seek, pause and tab changes. Keep user media
on the local machine. Do not commit without the user's instruction.
