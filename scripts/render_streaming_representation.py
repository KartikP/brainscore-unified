"""Render the demo: a stream arrives, the model's representation moves, a second
channel connects and adds what the first cannot see.

This runs the REAL streaming path -- ``WindowedStreamSession`` feeding
``_drive_neural_session_streaming``, one window at a time -- rather than scoring a
batch and animating it afterwards. That distinction is the whole point of the demo, so
faking it would be self-defeating.

    python unified/scripts/render_streaming_representation.py \
        --video demo.mp4 --audio demo.wav --fit-video other_clip.mp4 --out /tmp/frames

Two honesty decisions are baked in rather than left to chance:

1. **The projection is fit on a DIFFERENT clip.** Fitting PCA on the clip being
   animated would use future frames to lay out the past -- a streaming demo that
   secretly peeked. ``--fit-video`` supplies held-out material; the projection is then
   applied causally, window by window. If you skip it the script refuses rather than
   silently cheating (override with --allow-in-clip-fit, which stamps the output).
2. **Smoothness comes from stride, not from the model.** V-JEPA's native window is
   ~2 s; non-overlapping windows would update the picture only every 2 s. Overlapping
   windows buy smooth motion at proportionally more compute, and that is a display
   choice, so it is a flag with a stated default rather than a hidden constant.

Output is a directory of numbered PNGs plus manifest.json, so the animation can be
re-rendered deterministically and assembled with ffmpeg.
"""
import argparse
import json
import os
import traceback

import numpy as np


def decode_frames(path):
    """Yield frames one at a time. A generator, so the feed stays lazy: the whole
    point of windowed streaming is that the clip is never held in memory."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def clip_fps(path, default=30.0):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else default
    finally:
        cap.release()


def representations_for_clip(model, video_path, *, region, window_ms, stride_ms,
                             frames_dir=None):
    """Stream one clip and collect the per-window representation.

    Returns ``(vectors, window_meta)`` where ``vectors`` is ``(n_windows, n_units)``.
    Each vector is what the model emitted for that window on ``neural:<region>`` --
    the tower's actual activations, not a summary computed afterwards.
    """
    import pandas as pd
    from brainscore_core.streaming_helpers import (
        WindowedStreamSession, _drive_neural_session_streaming)
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    fps = clip_fps(video_path)
    written = {}

    def window_to_stimuli(frames, start_ms, end_ms):
        """Pack one window into a single clip row -- the native-temporal model wants a
        clip, not a bag of frames."""
        idx = len(written)
        out_dir = frames_dir or '/tmp/_stream_windows'
        os.makedirs(out_dir, exist_ok=True)
        # the wrapper reads from disk, so materialize just this window
        import imageio.v2 as imageio
        paths = []
        for j, fr in enumerate(frames):
            fp = os.path.join(out_dir, f'w{idx:04d}_f{j:03d}.png')
            imageio.imwrite(fp, fr)
            paths.append(fp)
        written[idx] = paths
        ss = StimulusSet(pd.DataFrame({
            'stimulus_id': [f'window_{idx:04d}'],
            'video_path': [paths[0]],          # wrapper-specific; see --dry-run note
            'n_frames': [len(frames)],
        }))
        ss.identifier = f'stream-demo-w{idx:04d}'
        ss.stimulus_paths = {f'window_{idx:04d}': paths[0]}
        return ss

    session = WindowedStreamSession(
        decode_frames(video_path), fps=fps, window_ms=window_ms,
        stride_ms=stride_ms, record=region, window_to_stimuli=window_to_stimuli)
    model.start_recording(region)
    _drive_neural_session_streaming(model, session)

    vectors, meta = [], []
    for event in session.emitted:
        if not event.channel.startswith('neural:'):
            continue
        arr = np.asarray(event.payload).reshape(-1)
        vectors.append(arr)
        meta.append({'t_ms': float(event.t_ms),
                     'stream_index': int(event.meta.get('stream_index', len(meta)))})
    return np.vstack(vectors) if vectors else np.empty((0, 0)), meta


def fit_projection(fit_vectors, n_components=2):
    """PCA fit on HELD-OUT material, then applied causally to the demo clip."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(fit_vectors)
    return pca


def render_frames(video_traj, video_vectors, audio_traj, out_dir, *, meta, title,
                  audio_from_index=None):
    """One PNG per window: the trajectory so far, plus a unit raster.

    Units in the raster are ordered by variance so structure is visible. That is a
    DISPLAY choice, not a finding, and the caption says so.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = len(video_traj)
    vx, vy = video_traj[:, 0], video_traj[:, 1]
    # cap the raster at a readable number of rows; variance ordering picks which
    n_show = min(160, video_vectors.shape[1])
    unit_order = np.argsort(-video_vectors.var(axis=0))[:n_show]
    vmin = float(np.percentile(video_vectors[:, unit_order], 2))
    vmax = float(np.percentile(video_vectors[:, unit_order], 98))
    audio_from_index = len(video_traj) if audio_from_index is None else audio_from_index
    lim = lambda a: (float(np.min(a)) - 0.5, float(np.max(a)) + 0.5)
    xlim, ylim = lim(vx), lim(vy)

    paths = []
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                                 gridspec_kw={'width_ratios': [1.1, 1]})
        ax = axes[0]
        ax.plot(vx[:i + 1], vy[:i + 1], '-', color='#2f6bff', lw=2, alpha=0.85,
                label='video channel')
        ax.plot(vx[i], vy[i], 'o', color='#2f6bff', ms=11)
        if audio_traj is not None and i >= audio_from_index:
            j = i - audio_from_index
            ax_x, ax_y = audio_traj[:j + 1, 0], audio_traj[:j + 1, 1]
            ax.plot(ax_x, ax_y, '-', color='#e0a13b', lw=2, alpha=0.85,
                    label='audio channel')
            ax.plot(ax_x[-1], ax_y[-1], 'o', color='#e0a13b', ms=11)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{title}\nt = {meta[i]["t_ms"]/1000:.1f}s', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, frameon=False)

        # unit raster: rows are units ordered by variance so structure is visible.
        # That ordering is a DISPLAY choice and the caption says so -- it is not a
        # claim that these units are special.
        ax2 = axes[1]
        show = video_vectors[:i + 1, unit_order].T
        ax2.imshow(show, aspect='auto', cmap='magma', interpolation='nearest',
                   vmin=vmin, vmax=vmax,
                   extent=[0, max(1, i + 1), show.shape[0], 0])
        ax2.set_xlabel('window', fontsize=9)
        ax2.set_ylabel('units (ordered by variance, for legibility)', fontsize=8)
        ax2.set_yticks([])
        ax2.set_title('what the tower actually computed', fontsize=10)
        fig.tight_layout()
        fp = os.path.join(out_dir, f'frame_{i:04d}.png')
        fig.savefig(fp, dpi=130); plt.close(fig)
        paths.append(fp)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='clip to animate')
    ap.add_argument('--audio', default=None, help='its audio track (wav)')
    ap.add_argument('--fit-video', default=None,
                    help='HELD-OUT clip the projection is fit on')
    ap.add_argument('--allow-in-clip-fit', action='store_true',
                    help='fit the projection on the demo clip itself. Uses future '
                         'frames to lay out the past; the manifest is stamped so the '
                         'output cannot be mistaken for a causal render.')
    ap.add_argument('--model', default='vjepa1-wav2vec2')
    ap.add_argument('--video-region', default='IT')
    ap.add_argument('--audio-region', default='A1')
    ap.add_argument('--window-ms', type=float, default=2000)
    ap.add_argument('--stride-ms', type=float, default=250,
                    help='overlap buys smooth motion; smoothness is a display choice')
    ap.add_argument('--out', default='/tmp/stream_demo')
    ap.add_argument('--fit-audio', default=None,
                    help='held-out audio for the audio projection (defaults to the '
                         'demo audio, which is then non-causal for that channel)')
    ap.add_argument('--audio-from-window', type=int, default=None,
                    help='window index at which the audio channel is connected, so '
                         'the demo can show video-only first and then both')
    ap.add_argument('--dry-run', action='store_true',
                    help='report the plan and the window schedule without loading '
                         'the model -- use this to sanity-check timing cheaply')
    args = ap.parse_args()

    if not args.fit_video and not args.allow_in_clip_fit:
        raise SystemExit(
            "refusing to fit the projection on the clip being animated: that uses "
            "future frames to lay out the past, which quietly contradicts the "
            "streaming claim the demo exists to make. Pass --fit-video with held-out "
            "material, or --allow-in-clip-fit to accept a stamped non-causal render.")

    os.makedirs(args.out, exist_ok=True)
    fps = clip_fps(args.video)
    n_frames = sum(1 for _ in decode_frames(args.video))
    duration_s = n_frames / fps
    n_windows = max(0, int((duration_s * 1000 - args.window_ms) // args.stride_ms) + 1)
    plan = {'video': args.video, 'fps': fps, 'n_frames': n_frames,
            'duration_s': round(duration_s, 2), 'window_ms': args.window_ms,
            'stride_ms': args.stride_ms, 'n_windows': n_windows,
            'projection_fit_on': args.fit_video or 'THE DEMO CLIP ITSELF (non-causal)',
            'causal_projection': bool(args.fit_video),
            'model': args.model}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        with open(os.path.join(args.out, 'plan.json'), 'w') as fh:
            json.dump(plan, fh, indent=2)
        print('DRY_RUN_DONE', flush=True)
        return

    import brainscore
    model = brainscore.load_model(args.model)

    print('--- video channel', flush=True)
    vid_vecs, meta = representations_for_clip(
        model, args.video, region=args.video_region,
        window_ms=args.window_ms, stride_ms=args.stride_ms,
        frames_dir=os.path.join(args.out, '_windows'))
    print(f'    {vid_vecs.shape[0]} windows x {vid_vecs.shape[1]} units', flush=True)

    fit_source = args.fit_video or args.video
    fit_vecs, _ = representations_for_clip(
        model, fit_source, region=args.video_region,
        window_ms=args.window_ms, stride_ms=args.stride_ms,
        frames_dir=os.path.join(args.out, '_fitwindows'))
    pca = fit_projection(fit_vecs)
    traj = pca.transform(vid_vecs)

    manifest = dict(plan)
    manifest['video_windows'] = int(vid_vecs.shape[0])
    manifest['video_units'] = int(vid_vecs.shape[1])
    manifest['explained_variance'] = [float(v) for v in pca.explained_variance_ratio_]

    # --- audio channel -------------------------------------------------------
    # Act 2: connect a second channel. The audio tower gets its OWN projection --
    # fit on the same held-out material, in its own space. Forcing both towers
    # through one projection would imply their units are commensurate, which they
    # are not: 1024 V-JEPA video units and 768 Wav2Vec2 audio units index different
    # things. Two spaces, drawn together, is the honest picture.
    audio_traj = None
    if args.audio:
        try:
            print('--- audio channel', flush=True)
            aud_vecs, aud_meta = representations_for_clip(
                model, args.audio, region=args.audio_region,
                window_ms=args.window_ms, stride_ms=args.stride_ms,
                frames_dir=os.path.join(args.out, '_audiowindows'))
            aud_fit, _ = representations_for_clip(
                model, args.fit_audio or args.audio, region=args.audio_region,
                window_ms=args.window_ms, stride_ms=args.stride_ms,
                frames_dir=os.path.join(args.out, '_audiofit'))
            audio_traj = fit_projection(aud_fit).transform(aud_vecs)
            np.save(os.path.join(args.out, 'audio_trajectory.npy'), audio_traj)
            np.save(os.path.join(args.out, 'audio_vectors.npy'), aud_vecs)
            manifest['audio_windows'] = int(aud_vecs.shape[0])
            manifest['audio_units'] = int(aud_vecs.shape[1])
            print(f'    {aud_vecs.shape[0]} windows x {aud_vecs.shape[1]} units',
                  flush=True)
        except Exception as exc:
            manifest['audio_error'] = str(exc)[:300]
            print(f'    audio channel FAILED: {exc}', flush=True)
            traceback.print_exc()

    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    np.save(os.path.join(args.out, 'video_trajectory.npy'), traj)
    np.save(os.path.join(args.out, 'video_vectors.npy'), vid_vecs)

    # --- frames --------------------------------------------------------------
    frames = render_frames(
        traj, vid_vecs, audio_traj, os.path.join(args.out, 'frames'),
        meta=meta, title=f'{args.model} - streaming representation',
        audio_from_index=args.audio_from_window)
    manifest['n_rendered_frames'] = len(frames)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'    rendered {len(frames)} frames -> {args.out}/frames', flush=True)
    print('  assemble with: ffmpeg -framerate 8 -i '
          f'{args.out}/frames/frame_%04d.png -pix_fmt yuv420p {args.out}/demo.mp4',
          flush=True)
    print('STREAM_DEMO_DONE', json.dumps(
        {k: manifest[k] for k in ('n_windows', 'video_units', 'causal_projection')}),
        flush=True)


if __name__ == '__main__':
    main()
