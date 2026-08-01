"""Three-panel streaming demo: the world, what the model was handed, what it computed.

Replaces the PCA-trajectory render. That version projected 1024-d video and 768-d audio
into 2-D, which forced three awkward caveats -- two incommensurate spaces drawn on one
axis, only ~25% of variance shown, and a projection that had to be fit on held-out
material to stay causal. None of that is needed to make the point, and all of it
invited misreading.

The point is the DISCRETIZATION: a continuous world arrives, the interface cuts it into
windows, each window is handed over as one event, and the model computes something for
each. So show exactly that.

    row 1  raw video, as a person would have seen it
    row 2  the timeline: a thumbnail dropped at every window the model was handed,
           on a video track, with an audio track appearing when that channel connects
    row 3  the activations, growing left to right as windows arrive

No projection, so nothing here is a shadow of something larger: row 3 is the units
themselves.

    python unified/scripts/render_streaming_panels.py \
        --video demo.mp4 --video-vectors video_vectors.npy \
        --audio-vectors audio_vectors.npy --audio-from-window 20 --out frames/
"""
import argparse
import json
import os

import numpy as np


def load_frames_at(video_path, times_s):
    """Grab one frame per requested time. Used for both the playhead and the strip."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wanted = {int(round(t * fps)): i for i, t in enumerate(times_s)}
    out = {}
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                out[wanted[idx]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
    return out, fps


def pool_if_flattened(vectors, n_units):
    """The saved video vectors flatten (tubelets x units); pool back over time.

    Flattening makes a window's vector depend on WHERE inside it content sits, so a
    75%-overlapping neighbour looks dissimilar (measured: r=0.34 flattened, 0.83
    pooled). Row 3 should show what the model represents, not that artifact.
    """
    if vectors.ndim == 2 and vectors.shape[1] % n_units == 0 and vectors.shape[1] != n_units:
        t = vectors.shape[1] // n_units
        return vectors.reshape(len(vectors), t, n_units).mean(axis=1)
    return vectors


def render_strips(video_path, strips, out_dir, *, window_ms, stride_ms,
                  n_windows, audio_from_window=None, caption='', n_units_strip=120,
                  dpi=130):
    """Row 3 as N labelled strips.

    ``strips`` is a list of ``(label, array, cmap, start_window)``. Two modes use it:
    channels (video + audio, the second appearing when it connects) and depths (three
    taps on the same tower). Same rows 1 and 2 either way -- what the model was handed
    does not change with which taps you read.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    os.makedirs(out_dir, exist_ok=True)
    n = n_windows
    starts_s = [i * stride_ms / 1000.0 for i in range(n)]
    total_s = starts_s[-1] + window_ms / 1000.0
    frames, fps = load_frames_at(video_path, starts_s)
    # single shared x range: the timeline needs a left margin for its row labels, so
    # the strips must use the SAME margin or a given instant lands at different x
    # positions in different rows and the panels stop being readable together.
    x_left = -total_s * 0.13
    thumb_every = max(1, n // 8)
    thumb_idx = list(range(0, n, thumb_every))

    prep = []
    for label, arr, cmap, start_w in strips:
        order = np.argsort(-arr.var(axis=0))[:n_units_strip]
        lo, hi = np.percentile(arr[:, order], [2, 98])
        prep.append((label, arr, cmap, start_w, order, lo, hi))

    paths = []
    for i in range(n):
        fig = plt.figure(figsize=(13, 4.2 + 1.5 * len(prep)))
        gs = fig.add_gridspec(2 + len(prep), 1,
                              height_ratios=[2.6, 1.7] + [1.0] * len(prep),
                              hspace=0.72)

        ax0 = fig.add_subplot(gs[0])
        fr = frames.get(i)
        if fr is not None:
            ax0.imshow(fr)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f'What a person sees   ·   t = {starts_s[i]:.1f}s', fontsize=13)

        ax1 = fig.add_subplot(gs[1])
        ax1.set_xlim(x_left, total_s); ax1.set_ylim(0, 1)
        ax1.set_yticks([]); ax1.set_xlabel('')
        ax1.set_title('What the model was handed: one window at a time', fontsize=12)
        ax1.hlines(0.68, 0, total_s, color='#dde3ee', lw=8, zorder=1)
        ax1.text(-total_s * 0.035, 0.68, 'video', ha='right', va='center',
                 fontsize=11, color='#2f6bff', fontweight='bold')
        for k in thumb_idx:
            if k > i:
                break
            th = frames.get(k)
            if th is None:
                continue
            small = th[::max(1, th.shape[0] // 44), ::max(1, th.shape[1] // 44)]
            ax1.add_artist(AnnotationBbox(
                OffsetImage(small, zoom=0.75), (starts_s[k], 0.68), frameon=True,
                pad=0.08, bboxprops=dict(edgecolor='#2f6bff', lw=1.2)))
        ax1.vlines([starts_s[k] for k in range(i + 1)], 0.58, 0.62,
                   color='#2f6bff', lw=1.4, zorder=3)
        if audio_from_window is not None and i >= audio_from_window:
            ax1.hlines(0.24, starts_s[audio_from_window], total_s,
                       color='#f2e3c4', lw=8, zorder=1)
            ax1.text(-total_s * 0.035, 0.24, 'audio', ha='right', va='center',
                     fontsize=11, color='#e0a13b', fontweight='bold')
            ax1.vlines([starts_s[k] for k in range(audio_from_window, i + 1)],
                       0.14, 0.18, color='#e0a13b', lw=1.4, zorder=3)
            ax1.annotate('audio channel connects here',
                         (starts_s[audio_from_window], 0.06), fontsize=10,
                         color='#e0a13b', ha='left', style='italic')
        ax1.axvline(starts_s[i], color='#d8483b', lw=1.6, zorder=4)

        for row, (label, arr, cmap, start_w, order, lo, hi) in enumerate(prep):
            ax = fig.add_subplot(gs[2 + row])
            if i >= start_w:
                strip = arr[start_w:i + 1, order].T
                ax.imshow(strip, aspect='auto', cmap=cmap, vmin=lo, vmax=hi,
                          interpolation='nearest',
                          extent=[starts_s[start_w],
                                  starts_s[i] + stride_ms / 1000.0, strip.shape[0], 0])
            ax.set_xlim(x_left, total_s); ax.set_yticks([])
            ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title('What the model computed', fontsize=12, pad=14)
            if row == len(prep) - 1:
                ax.set_xlabel(caption or
                              'time (s)   ·   units ordered by variance, for legibility',
                              fontsize=9)
            else:
                ax.set_xticks([])

        fp = os.path.join(out_dir, f'panel_{i:04d}.png')
        fig.savefig(fp, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        paths.append(fp)
    return paths


def render(video_path, vid_vecs, aud_vecs, out_dir, *, window_ms, stride_ms,
           audio_from_window, n_units_strip=120, dpi=130):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    os.makedirs(out_dir, exist_ok=True)
    n = len(vid_vecs)
    starts_s = [i * stride_ms / 1000.0 for i in range(n)]
    total_s = starts_s[-1] + window_ms / 1000.0

    frames, fps = load_frames_at(video_path, starts_s)

    # thumbnails are sparse: one every `thumb_every` windows, or the strip is a smear
    thumb_every = max(1, n // 8)
    thumb_idx = list(range(0, n, thumb_every))

    # order units by variance for legibility (a DISPLAY choice, stated in the caption)
    v_order = np.argsort(-vid_vecs.var(axis=0))[:n_units_strip]
    a_order = np.argsort(-aud_vecs.var(axis=0))[:n_units_strip] if aud_vecs is not None else None
    v_lo, v_hi = np.percentile(vid_vecs[:, v_order], [2, 98])
    if aud_vecs is not None:
        a_lo, a_hi = np.percentile(aud_vecs[:, a_order], [2, 98])

    paths = []
    for i in range(n):
        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.7, 2.2], hspace=0.55)

        # --- row 1: the world -------------------------------------------------
        ax0 = fig.add_subplot(gs[0])
        fr = frames.get(i)
        if fr is not None:
            ax0.imshow(fr)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f'What a person sees   ·   t = {starts_s[i]:.1f}s', fontsize=13)

        # --- row 2: what the model was handed ---------------------------------
        ax1 = fig.add_subplot(gs[1])
        ax1.set_xlim(x_left, total_s); ax1.set_ylim(0, 1)
        ax1.set_yticks([]); ax1.set_xlabel('')
        ax1.set_title('What the model was handed: one window at a time', fontsize=12)
        # video track
        ax1.hlines(0.68, 0, total_s, color='#dde3ee', lw=8, zorder=1)
        ax1.text(-total_s * 0.035, 0.68, 'video', ha='right', va='center',
                 fontsize=11, color='#2f6bff', fontweight='bold')
        for k in thumb_idx:
            if k > i:
                break
            th = frames.get(k)
            if th is None:
                continue
            small = th[::max(1, th.shape[0] // 44), ::max(1, th.shape[1] // 44)]
            ab = AnnotationBbox(OffsetImage(small, zoom=0.75), (starts_s[k], 0.68),
                                frameon=True, pad=0.08,
                                bboxprops=dict(edgecolor='#2f6bff', lw=1.2))
            ax1.add_artist(ab)
        # every window that has arrived, as a tick -- the thumbnails are only a sample
        ax1.vlines([starts_s[k] for k in range(i + 1)], 0.58, 0.62,
                   color='#2f6bff', lw=1.4, zorder=3)
        # audio track, appears only once that channel is connected
        if aud_vecs is not None and i >= audio_from_window:
            ax1.hlines(0.24, starts_s[audio_from_window], total_s,
                       color='#f2e3c4', lw=8, zorder=1)
            ax1.text(-total_s * 0.035, 0.24, 'audio', ha='right', va='center',
                     fontsize=11, color='#e0a13b', fontweight='bold')
            ax1.vlines([starts_s[k] for k in range(audio_from_window, i + 1)],
                       0.14, 0.18, color='#e0a13b', lw=1.4, zorder=3)
            ax1.annotate('audio channel connects here',
                         (starts_s[audio_from_window], 0.06),
                         fontsize=10, color='#e0a13b', ha='left', style='italic')
        ax1.axvline(starts_s[i], color='#d8483b', lw=1.6, zorder=4)

        # --- row 3: what it computed ------------------------------------------
        ax2 = fig.add_subplot(gs[2])
        strip = vid_vecs[:i + 1, v_order].T
        ax2.imshow(strip, aspect='auto', cmap='magma', vmin=v_lo, vmax=v_hi,
                   interpolation='nearest',
                   extent=[0, starts_s[i] + stride_ms / 1000.0, strip.shape[0], 0])
        ax2.set_xlim(0, total_s)
        ax2.set_ylabel('video units', fontsize=10)
        ax2.set_yticks([])
        ax2.set_xlabel('time (s)   ·   units ordered by variance, for legibility',
                       fontsize=9)
        ax2.set_title('What the model computed', fontsize=12)
        if aud_vecs is not None and i >= audio_from_window:
            div = ax2.inset_axes([0, -0.62, 1, 0.5])
            a_strip = aud_vecs[audio_from_window:i + 1, a_order].T
            div.imshow(a_strip, aspect='auto', cmap='cividis', vmin=a_lo, vmax=a_hi,
                       interpolation='nearest',
                       extent=[starts_s[audio_from_window],
                               starts_s[i] + stride_ms / 1000.0, a_strip.shape[0], 0])
            div.set_xlim(0, total_s); div.set_yticks([]); div.set_xticks([])
            div.set_ylabel('audio units', fontsize=10)

        fp = os.path.join(out_dir, f'panel_{i:04d}.png')
        fig.savefig(fp, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        paths.append(fp)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--video-vectors', required=True)
    ap.add_argument('--audio-vectors', default=None)
    ap.add_argument('--video-units', type=int, default=1024)
    ap.add_argument('--window-ms', type=float, default=2000)
    ap.add_argument('--stride-ms', type=float, default=500)
    ap.add_argument('--audio-from-window', type=int, default=20)
    ap.add_argument('--out', default='/tmp/panels')
    args = ap.parse_args()

    vid = pool_if_flattened(np.load(args.video_vectors), args.video_units)
    aud = np.load(args.audio_vectors) if args.audio_vectors else None
    print(f'video {vid.shape}  audio {None if aud is None else aud.shape}', flush=True)
    paths = render(args.video, vid, aud, args.out,
                   window_ms=args.window_ms, stride_ms=args.stride_ms,
                   audio_from_window=args.audio_from_window)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump({'n_panels': len(paths), 'video_units': int(vid.shape[1]),
                   'audio_units': None if aud is None else int(aud.shape[1]),
                   'window_ms': args.window_ms, 'stride_ms': args.stride_ms,
                   'audio_from_window': args.audio_from_window,
                   'projection': 'none - units shown directly'}, fh, indent=2)
    print(f'rendered {len(paths)} panels -> {args.out}', flush=True)
    print('assemble: ffmpeg -framerate 6 -i '
          f'{args.out}/panel_%04d.png -pix_fmt yuv420p {args.out}/panels.mp4', flush=True)


if __name__ == '__main__':
    main()
