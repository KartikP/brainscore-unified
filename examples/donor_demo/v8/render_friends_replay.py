#!/usr/bin/env python
"""Moving-brain replay from the Friends (Courtois NeuroMod / Algonauts 2025) held-out predictions.

Four panels, 1680x480, 30 fps: Watch (the episode segment at native frame rate, with the words
heard in each 1.49 s scan), Record (CLIP video / Wav2Vec2 sound / MiniLM words features for the
current scan), Predict (held-out ridge prediction, video+sound+words) and Compare (measured BOLD,
participant 1). Brain maps are Schaefer-1000 parcels on fsaverage5 (left hemisphere, lateral and
medial) and step once per scan (TR = 1.49 s), which is the resolution of the recording.
"""
import argparse, ast, csv, json, subprocess, sys
from pathlib import Path
import numpy as np, cv2, nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw, ImageFont

PAPER = (246, 245, 241); INK = (20, 20, 20); MUTED = (110, 110, 105)
DIVERGING = LinearSegmentedColormap.from_list('bs_div', ['#2a78d6', '#f0efec', '#e34948'])
SEQ = {'video': LinearSegmentedColormap.from_list('v', ['#f0efec', '#2a78d6']),
       'audio': LinearSegmentedColormap.from_list('a', ['#f0efec', '#eb6834']),
       'text': LinearSegmentedColormap.from_list('t', ['#f0efec', '#1baf7a'])}
TR = 1.49

def font(size, bold=False):
    try: return ImageFont.truetype('/System/Library/Fonts/HelveticaNeue.ttc', size, index=1 if bold else 0)
    except Exception: return ImageFont.load_default()

def _autocrop(img, bg, margin=6):
    arr = np.asarray(img).astype(int); diff = np.abs(arr - np.array(bg)).sum(2) > 18
    ys, xs = np.where(diff)
    if len(xs) == 0: return img
    return img.crop((max(0, xs.min() - margin), max(0, ys.min() - margin), min(img.width, xs.max() + margin), min(img.height, ys.max() + margin)))

def render_brain(mesh, values_vertex, vmax, view, out_px):
    from nilearn import plotting
    coords, faces, sulc = mesh
    fig = plt.figure(figsize=(8, 6), dpi=100); ax = fig.add_subplot(111, projection='3d')
    plotting.plot_surf_stat_map([coords, faces], values_vertex, bg_map=sulc, view=view, cmap=DIVERGING, vmax=vmax,
                                symmetric_cbar=True, colorbar=False, threshold=None, axes=ax, figure=fig, hemi='left')
    paper = np.array(PAPER) / 255; ax.set_facecolor(paper); fig.patch.set_facecolor(paper); fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw(); img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
    img = _autocrop(img, PAPER); scale = min(out_px[0] / img.width, out_px[1] / img.height)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    canvas = Image.new('RGB', out_px, PAPER); canvas.paste(img, ((out_px[0] - img.width) // 2, (out_px[1] - img.height) // 2))
    return canvas

def strip(vec, lo, hi, cmap, shape, px_w):
    g = vec.reshape(shape); rgba = (cmap(Normalize(lo, hi)(g)) * 255).astype(np.uint8)[:, :, :3]
    return Image.fromarray(rgba).resize((px_w, int(px_w * shape[0] / shape[1])), Image.NEAREST)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--friends', required=True, help='folder with heldout_pred__sub01.npz')
    ap.add_argument('--demo', required=True, help='folder with the episode segment, features, transcript')
    ap.add_argument('--annot', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', type=int, default=154); ap.add_argument('--tr0', type=int, default=297); ap.add_argument('--n_tr', type=int, default=40)
    ap.add_argument('--seg_start_s', type=float, default=430.0); ap.add_argument('--episode', default='friends_s04e05a')
    ap.add_argument('--condition', default='video+audio+text'); ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()
    O = Path(args.out); (O / 'frames').mkdir(parents=True, exist_ok=True); D = Path(args.demo)
    z = np.load(Path(args.friends) / 'heldout_pred__sub01.npz')
    run_idx = z['run_idx']; t_all = z['t_within_run']; rows = np.flatnonzero(run_idx == args.run)
    Yall = z['Y'].astype(np.float32); Pall = z[f'pred__{args.condition}'].astype(np.float32)
    mu = Yall.mean(0); sd = Yall.std(0) + 1e-6          # one display transform for measured and predicted
    tr_of_row = {int(t_all[i]): i for i in rows}
    trs = list(range(args.tr0, args.tr0 + args.n_tr))
    assert all(t in tr_of_row for t in trs), 'window outside held-out rows'
    Zm = {t: (Yall[tr_of_row[t]] - mu) / sd for t in trs}; Zp = {t: (Pall[tr_of_row[t]] - mu) / sd for t in trs}
    def spat(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
    r_tr = {t: spat(Yall[tr_of_row[t]], Pall[tr_of_row[t]]) for t in trs}
    VMAX = 2.0
    # parcels -> left-hemisphere vertices
    lab, _, _ = nib.freesurfer.read_annot(args.annot)
    def to_vertex(vals1000):
        v = np.full(lab.shape, np.nan, np.float32); m = lab > 0; v[m] = vals1000[lab[m] - 1]; return v
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage('fsaverage5')
    coords, faces = surface.load_surf_mesh(fs['infl_left']); sulc = surface.load_surf_data(fs['sulc_left']); mesh = (coords, faces, sulc)
    # features for the Record panel
    Fv = np.load(D / 'clip_video_s04e05a.npz'); fv_t = {int(t): x.astype(np.float32) for t, x in zip(Fv['t'], Fv['X'])}
    fa = np.load(D / 'audio_features_wav2vec2_s04e05a.npy').astype(np.float32); ft = np.load(D / 'text_features_minilm_s04e05a.npy').astype(np.float32)
    feats = {'video': {t: fv_t[t] for t in trs}, 'audio': {t: fa[t] for t in trs}, 'text': {t: ft[t] for t in trs}}
    lims = {k: (np.percentile(np.stack(list(v.values())), 2), np.percentile(np.stack(list(v.values())), 98)) for k, v in feats.items()}
    shapes = {'video': (12, 64), 'audio': (12, 64), 'text': (6, 64)}
    # transcript: words heard in each TR
    words = {}
    with open(D / f'{args.episode}.tsv') as fh:
        rd = csv.reader(fh, delimiter='\t'); next(rd)
        for i, row in enumerate(rd):
            words[i] = row[0].strip() if row and row[0] else ''
    # video
    cap = cv2.VideoCapture(str(D / 's04e05a_430_515.mp4')); nat_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    t_start = args.tr0 * TR; off0 = t_start - args.seg_start_s
    assert off0 >= 0 and (off0 + args.n_tr * TR) * nat_fps < len(frames), 'segment does not cover window'
    Wpx, Hpx = 1680, 480; pad = 24; col_w = (Wpx - 5 * pad) // 4; top = 12; panel_h = Hpx - top - 60
    x0 = [pad + i * (col_w + pad) for i in range(4)]; brain_px = (col_w, panel_h // 2)
    f_small = font(17); f_num = font(26, bold=True); f_mid = font(17); f_words = font(19)
    brains = {}
    for t in trs:
        brains[t] = {'pred_lat': render_brain(mesh, to_vertex(Zp[t]), VMAX, 'lateral', brain_px), 'pred_med': render_brain(mesh, to_vertex(Zp[t]), VMAX, 'medial', brain_px),
                     'meas_lat': render_brain(mesh, to_vertex(Zm[t]), VMAX, 'lateral', brain_px), 'meas_med': render_brain(mesh, to_vertex(Zm[t]), VMAX, 'medial', brain_px)}
        print('brain renders for TR', t, flush=True)
    json.dump({'episode': args.episode, 'participant': 'sub-01', 'run_index': args.run, 'TR_s': TR, 'window_TRs': [args.tr0, args.tr0 + args.n_tr - 1],
               'window_episode_time_s': [t_start, t_start + args.n_tr * TR], 'condition': args.condition,
               'z_scoring': 'per-parcel (measured mean, measured SD over all held-out rows of participant 1), applied to measured and predicted alike',
               'color_range_z': [-VMAX, VMAX], 'colormap': 'blue(#2a78d6) - grey(#f0efec) - red(#e34948)', 'parcels': 'Schaefer 2018, 1000 parcels, 7 networks (fsaverage5 annot)',
               'views': ['left hemisphere lateral', 'left hemisphere medial'], 'hemodynamic_lag_note': 'the fMRI response at a scan reflects the picture ~4-6 s earlier; the mapping uses the 5 preceding scans with a 3-scan delay',
               'selection_rule': 'participant 1; held-out run with the highest median per-parcel temporal r (video+audio+text); the 40-TR window within it with the highest mean per-TR pattern r',
               'per_tr_pattern_r': {str(t): r_tr[t] for t in trs}}, open(O / 'display_params.json', 'w'), indent=1)
    fps = args.fps; n_out = int(round(args.n_tr * TR * fps)); fi = 0; rows_out = []
    for j in range(n_out):
        tw = j / fps; t = args.tr0 + int(tw // TR); t = min(t, trs[-1])
        canvas = Image.new('RGB', (Wpx, Hpx), PAPER); dr = ImageDraw.Draw(canvas)
        fr = Image.fromarray(frames[min(int((off0 + tw) * nat_fps), len(frames) - 1)])
        fr = fr.resize((col_w, int(col_w * fr.height / fr.width)))
        fy = top + 10; canvas.paste(fr, (x0[0], fy))
        w = words.get(t, '')
        dr.text((x0[0], fy + fr.height + 10), f'heard: "{w}"' if w else 'heard: (no speech)', fill=INK, font=f_words)
        dr.text((x0[0], Hpx - 44), f'Friends S4 E5 · {t_start + tw:0.1f} s · scan {t - args.tr0 + 1} of {args.n_tr}', fill=MUTED, font=f_small)
        # Record: three feature strips for the current scan
        y = top + 6
        for k, label in (('video', 'video model (CLIP)'), ('audio', 'sound model (Wav2Vec2)'), ('text', 'word model (MiniLM)')):
            im = strip(feats[k][t], *lims[k], SEQ[k], shapes[k], col_w); canvas.paste(im, (x0[1], y))
            dr.text((x0[1], y + im.height + 2), label, fill=MUTED, font=f_small); y += im.height + 30
        dr.text((x0[1], Hpx - 44), 'one feature vector per 1.49 s scan', fill=MUTED, font=f_small)
        for i, key in ((2, 'pred'), (3, 'meas')):
            canvas.paste(brains[t][key + '_lat'], (x0[i], top)); canvas.paste(brains[t][key + '_med'], (x0[i], top + brain_px[1]))
        dr.text((x0[2], Hpx - 44), 'held-out episode · mapping fitted on other episodes', fill=MUTED, font=f_small)
        dr.text((x0[3], Hpx - 50), f'match r = {r_tr[t]:0.2f}', fill=INK, font=f_num)
        dr.text((x0[3] + 190, Hpx - 42), 'participant 1 · ~5 s lag', fill=MUTED, font=f_mid)
        canvas.save(O / 'frames' / f'f{fi:05d}.png')
        rows_out.append([fi, round(tw, 3), t, round(t_start + tw, 3), words.get(t, ''), round(r_tr[t], 4)]); fi += 1
    with open(O / 'frame_map.csv', 'w', newline='') as fcsv:
        wr = csv.writer(fcsv); wr.writerow(['frame', 'wall_time_s', 'TR', 'episode_time_s', 'words_heard', 'pattern_r_video+audio+text']); wr.writerows(rows_out)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(O / 'frames' / 'f%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(O / 'replay_friends.mp4')], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(O / 'replay_friends.mp4'), '-ss', f'{off0:.3f}', '-t', f'{args.n_tr * TR:.3f}', '-i', str(D / 's04e05a_430_515.mp4'),
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-shortest', str(O / 'replay_friends_with_sound.mp4')], check=True)
    print('wrote', fi, 'frames', fi / fps, 's')

if __name__ == '__main__':
    main()
