"""Render the measured replay ("watch, record, predict, compare") as frames + MP4.

Inputs are the saved experiment outputs (no inference here):
  <data>/demo/selection.json, <data>/demo/frame_meta.json, <data>/demo/<clip>/frame_XX.jpg,
  <data>/demo/<clip>/tokens_blocks16.npy          (8, 196, 1024) float16, V-JEPA v1 block 16
  <data>/demo/clips/<clip>.mp4                      the stimulus itself (copied, for playback)
  <data>/neural/targets.npz                          Y_mean (clips x vertices), masks, ceiling
  <data>/predictions/heldout_pred__mean__video__whole.npy  held-out predictions (clips x vertices)
  <data>/metrics/per_clip_spatial_r__mean__video__visual.npy
  <data>/features/pooled_features.npz

Every rendered frame is logged to frame_map.csv: frame index, wall time, clip id,
clip time, which tubelet (model time bin) is shown, and whether the brain maps are
visible. Display normalization parameters are written to display_params.json.

Cadences are deliberately independent:
  stimulus  : native clip frame rate (decoded from the MP4)
  model     : one update per V-JEPA tubelet (8 per 3 s clip = 375 ms)
  brain     : one measured response per clip (GLM beta), shown once the clip ends
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw, ImageFont

# ---- palette (validated with the dataviz validator; see methods doc) ----
PAPER = (246, 245, 241)
INK = (20, 20, 20)
MUTED = (110, 110, 105)
VIDEO_BLUE = '#2a78d6'
DIVERGING = LinearSegmentedColormap.from_list('bs_div', ['#2a78d6', '#f0efec', '#e34948'])
SEQ_GREY = LinearSegmentedColormap.from_list('bs_seq', ['#f0efec', '#184f95'])


def font(size, bold=False):
    for cand in (['/System/Library/Fonts/HelveticaNeue.ttc'] if True else []):
        try:
            return ImageFont.truetype(cand, size, index=1 if bold else 0)
        except Exception:
            pass
    return ImageFont.load_default()


def hemi_mesh(fs, hemi):
    from nilearn import surface
    coords, faces = surface.load_surf_mesh(fs[f'infl_{hemi}'])
    sulc = surface.load_surf_data(fs[f'sulc_{hemi}'])
    return coords, faces, sulc


def _autocrop(img, bg, margin=6):
    arr = np.asarray(img).astype(int)
    diff = np.abs(arr - np.array(bg)).sum(2) > 18
    ys, xs = np.where(diff)
    if len(xs) == 0:
        return img
    box = (max(0, xs.min() - margin), max(0, ys.min() - margin), min(img.width, xs.max() + margin), min(img.height, ys.max() + margin))
    return img.crop(box)


def render_brain(mesh, values_hemi, mask_hemi, vmax, view, out_px):
    """One hemisphere. values/mask are (10242,). Non-ROI vertices show grey sulcal shading.
    Rendered large, auto-cropped to the brain, then fitted into out_px on paper."""
    from nilearn import plotting
    coords, faces, sulc = mesh
    data = np.where(mask_hemi, values_hemi, np.nan)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    plotting.plot_surf_stat_map([coords, faces], data, bg_map=sulc, view=view, cmap=DIVERGING,
                                vmax=vmax, symmetric_cbar=True, colorbar=False, threshold=None,
                                axes=ax, figure=fig, hemi='left')
    paper = np.array(PAPER) / 255
    ax.set_facecolor(paper); fig.patch.set_facecolor(paper)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    img = _autocrop(img, PAPER)
    scale = min(out_px[0] / img.width, out_px[1] / img.height)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    canvas = Image.new('RGB', out_px, PAPER)
    canvas.paste(img, ((out_px[0] - img.width) // 2, (out_px[1] - img.height) // 2))
    return canvas


def token_grid_image(tokens_t, vmin, vmax, px):
    """tokens_t: (196,) scalar per spatial patch (14x14) -> image."""
    g = tokens_t.reshape(14, 14)
    n = Normalize(vmin, vmax)
    rgba = (SEQ_GREY(n(g)) * 255).astype(np.uint8)[:, :, :3]
    img = Image.fromarray(rgba).resize((px, px), Image.NEAREST)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--hold_sec', type=float, default=2.2)
    ap.add_argument('--n_clips', type=int, default=10)
    ap.add_argument('--size', default='1680x480')
    args = ap.parse_args()
    D = Path(args.data); O = Path(args.out); O.mkdir(parents=True, exist_ok=True)
    (O / 'frames').mkdir(exist_ok=True)
    Wpx, Hpx = map(int, args.size.split('x'))

    sel = json.load(open(D / 'demo' / 'selection.json'))
    meta = json.load(open(D / 'demo' / 'frame_meta.json'))
    clips = sel.get('display', sel['top'])[:args.n_clips]
    tg = np.load(D / 'neural' / 'targets.npz', allow_pickle=True)
    clip_ids = [str(c) for c in tg['clip_ids']]
    Y = tg['Y_mean'].astype(np.float32)
    vis = tg['visual_roi_mask'].astype(bool)
    P = np.load(D / 'predictions' / 'heldout_pred__mean__video__whole.npy').astype(np.float32)
    sp = np.load(D / 'metrics' / 'per_clip_spatial_r__mean__video__visual.npy')

    # ---- display normalization: ONE transform for measured and predicted ----
    # per-vertex z-scoring with the measured mean/SD across all 1026 clips; the same
    # (mean, SD) is applied to predictions so colours are comparable across panels and clips.
    mu = Y.mean(0); sd = Y.std(0) + 1e-6
    Z = (Y - mu) / sd; ZP = (P - mu) / sd
    VMAX = 2.0
    json.dump({'z_scoring': 'per-vertex (measured mean, measured SD over 1026 clips), applied to measured and predicted alike',
               'color_range_z': [-VMAX, VMAX], 'colormap': 'blue(#2a78d6) - grey(#f0efec) - red(#e34948), neutral midpoint = clip-average response',
               'vertices_shown': 'visual ROI (split-half reliability >= 0.3, 4042 vertices); other cortex drawn as grey sulcal shading',
               'views': ['left hemisphere lateral', 'left hemisphere medial'], 'surface': 'fsaverage5 inflated (nilearn)'},
              open(O / 'display_params.json', 'w'), indent=1)

    from nilearn import datasets
    fs = datasets.fetch_surf_fsaverage('fsaverage5')
    mesh_l = hemi_mesh(fs, 'left')

    # layout (pixels) inside the replay canvas
    pad = 24
    col_w = (Wpx - 5 * pad) // 4
    top = 12
    panel_h = Hpx - top - 60
    x0 = [pad + i * (col_w + pad) for i in range(4)]
    f_small = font(17); f_num = font(26, bold=True); f_mid = font(17)

    brain_px = (col_w, panel_h // 2)
    fps = args.fps
    frame_rows = []
    fi = 0
    cache_brain = {}
    for ci, cid in enumerate(clips):
        m = meta[cid]
        k = clip_ids.index(cid)
        # stimulus frames at native rate
        cap = cv2.VideoCapture(str(D / 'demo' / 'clips' / f'{cid}.mp4'))
        nat_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        vid_frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            vid_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        dur = len(vid_frames) / nat_fps
        tokens = np.load(D / 'demo' / cid / 'tokens_blocks16.npy').astype(np.float32)  # (8,196,1024)
        # scalar per token: projection on the clip's first principal direction of block-16 tokens
        flat = tokens.reshape(-1, tokens.shape[-1])
        u, s, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
        proj = ((flat - flat.mean(0)) @ vt[0]).reshape(8, 196)
        tmin, tmax = np.percentile(proj, 2), np.percentile(proj, 98)
        windows = m['tubelet_time_windows_s']
        # brain renders (once per clip)
        if cid not in cache_brain:
            L = slice(0, 10242)
            cache_brain[cid] = {
                'pred_post': render_brain(mesh_l, ZP[k][L], vis[L], VMAX, 'lateral', brain_px),
                'pred_vent': render_brain(mesh_l, ZP[k][L], vis[L], VMAX, 'medial', brain_px),
                'meas_post': render_brain(mesh_l, Z[k][L], vis[L], VMAX, 'lateral', brain_px),
                'meas_vent': render_brain(mesh_l, Z[k][L], vis[L], VMAX, 'medial', brain_px),
            }
        total_sec = dur + args.hold_sec
        n_out = int(round(total_sec * fps))
        for j in range(n_out):
            t = j / fps
            canvas = Image.new('RGB', (Wpx, Hpx), PAPER)
            dr = ImageDraw.Draw(canvas)
            # --- Watch: native-rate stimulus, then hold last frame dimmed
            vi = min(int(t * nat_fps), len(vid_frames) - 1)
            fr = Image.fromarray(vid_frames[vi])
            fr = fr.resize((col_w, int(col_w * fr.height / fr.width)))
            if t > dur:
                fr = Image.blend(fr, Image.new('RGB', fr.size, PAPER), 0.35)
            canvas.paste(fr, (x0[0], top + (panel_h - fr.height) // 2))
            dr.text((x0[0], Hpx - 44), f'clip {cid.replace("stimulus_", "")}  ·  {min(t, dur):0.1f} / {dur:0.1f} s', fill=MUTED, font=f_small)
            # --- Record: token grid for the tubelet whose window contains t (holds last after clip end)
            tb = 7
            for q, (a, b) in enumerate(windows):
                if t <= b:
                    tb = q
                    break
            grid = token_grid_image(proj[tb], tmin, tmax, col_w)
            gy = top + (panel_h - col_w) // 2
            canvas.paste(grid, (x0[1], gy))
            # tubelet progress ticks
            for q in range(8):
                col = (24, 24, 24) if q <= tb else (205, 204, 198)
                dr.rectangle([x0[1] + q * (col_w // 8) + 2, gy + col_w + 10, x0[1] + (q + 1) * (col_w // 8) - 2, gy + col_w + 16], fill=col)
            dr.text((x0[1], Hpx - 44), f'V-JEPA block 16 · 14×14 patches · step {tb + 1} of 8', fill=MUTED, font=f_small)
            # --- Predict / Compare: appear after the clip ends (one response per clip)
            show = t >= dur
            if show:
                for i, key in ((2, 'pred'), (3, 'meas')):
                    canvas.paste(cache_brain[cid][key + '_post'], (x0[i], top))
                    canvas.paste(cache_brain[cid][key + '_vent'], (x0[i], top + brain_px[1]))
                dr.text((x0[2], Hpx - 44), 'held-out clip · map fitted on other clips', fill=MUTED, font=f_small)
                dr.text((x0[3], Hpx - 50), f'match r = {sp[k]:0.2f}', fill=INK, font=f_num)
                dr.text((x0[3] + 190, Hpx - 42), '10 participants, averaged', fill=MUTED, font=f_mid)
            else:
                for i in (2, 3):
                    dr.rectangle([x0[i], top, x0[i] + col_w, top + panel_h], outline=(215, 214, 208), width=2)
                dr.text((x0[2], Hpx - 44), 'appears when the clip ends', fill=MUTED, font=f_small)
                dr.text((x0[3], Hpx - 44), 'one fMRI response per clip', fill=MUTED, font=f_small)
            canvas.save(O / 'frames' / f'f{fi:05d}.png')
            frame_rows.append([fi, round(fi / fps, 3), cid, round(min(t, dur), 3), tb + 1, int(show), round(float(sp[k]), 4)])
            fi += 1
        print(cid, 'frames so far', fi, flush=True)
    with open(O / 'frame_map.csv', 'w', newline='') as fcsv:
        w = csv.writer(fcsv); w.writerow(['frame', 'wall_time_s', 'clip_id', 'clip_time_s', 'model_time_bin', 'brain_visible', 'clip_spatial_r_video_visualROI'])
        w.writerows(frame_rows)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(O / 'frames' / 'f%05d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(O / 'replay.mp4')], check=True)
    print('wrote', O / 'replay.mp4', 'frames', fi, 'duration', fi / fps)


if __name__ == '__main__':
    main()
