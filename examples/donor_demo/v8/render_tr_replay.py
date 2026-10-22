#!/usr/bin/env python
"""Moving-brain replay from the single-trial (per-TR) BOLDMoments recording, participant 1.
Watch: clips at native rate as presented in the scanner (3 s clip, 1 s blank); Record: V-JEPA block-16
clip-average activity (1024 units as a 32x32 grid); Predict / Compare: held-out per-TR prediction vs
measured BOLD (z-scored per run), visual ROI, left hemisphere, stepping every TR = 1.75 s."""
import argparse, csv, json, subprocess
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw, ImageFilter
import sys; sys.path.insert(0, str(Path(__file__).parent))
from render_friends_replay import font, render_brain, PAPER, INK, MUTED
SEQ_GREY = LinearSegmentedColormap.from_list('bs_seq', ['#f0efec', '#184f95'])
EXCLUDE = {'stimulus_1048', 'stimulus_1058', 'stimulus_1002'}

def pick_window(events, demo, win_s):
    on = [(e['onset_sec'], e['stimulus_id']) for e in events]
    best = None
    for o0, _ in on:
        s0 = o0 - 1.0
        inside = [sid for o, sid in on if s0 <= o < s0 + win_s - 3.5]
        n_demo = sum(sid in demo for sid in inside); n_bad = sum(sid in EXCLUDE for sid in inside)
        key = (n_bad == 0, n_demo, -s0)
        if best is None or key > best[0]: best = (key, s0, inside)
    return best[1], best[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_npz', required=True); ap.add_argument('--data', required=True, help='Experiment A package (neural/targets.npz, features/pooled_features.npz, demo/selection.json)')
    ap.add_argument('--clips', required=True, help='folder with <stimulus_id>.mp4 for the window'); ap.add_argument('--out', required=True)
    ap.add_argument('--win_s', type=float, default=63.0); ap.add_argument('--list_clips', action='store_true'); ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()
    D = Path(args.data); z = np.load(args.run_npz, allow_pickle=True); events = json.loads(str(z['events'])); TR = float(z['tr_sec'])
    demo = set(json.load(open(D / 'demo' / 'selection.json'))['display'])
    s0, inside = pick_window(events, demo, args.win_s)
    if args.list_clips: print(' '.join(inside)); print('window_start', s0, file=sys.stderr); return
    O = Path(args.out); (O / 'frames').mkdir(parents=True, exist_ok=True)
    Ym = z['measured_z'].astype(np.float32); Pm = z['predicted'].astype(np.float32); sr = z['spatial_r_roi'].astype(np.float32)
    tg = np.load(D / 'neural' / 'targets.npz', allow_pickle=True); vis = tg['visual_roi_mask'].astype(bool)
    f = np.load(D / 'features' / 'pooled_features.npz', allow_pickle=True); feat = {str(c): v for c, v in zip(f['clip_ids'], f['video'])}
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage('fsaverage5'); coords, faces = surface.load_surf_mesh(fs['infl_left']); sulc = surface.load_surf_data(fs['sulc_left']); mesh = (coords, faces, sulc)
    VMAX = 2.0; L = slice(0, 10242)
    def vals(v): a = np.where(vis[L], v[L], np.nan); return a
    tr0 = int(np.floor(s0 / TR)); n_tr = int(np.ceil(args.win_s / TR)) + 1
    Wpx, Hpx = 1680, 480; pad = 24; col_w = (Wpx - 5 * pad) // 4; top = 12; panel_h = Hpx - top - 60
    x0 = [pad + i * (col_w + pad) for i in range(4)]; brain_px = (col_w, panel_h // 2)
    brains = {}
    for t in range(tr0, min(tr0 + n_tr, len(Ym))):
        brains[t] = {'pred_lat': render_brain(mesh, vals(Pm[t]), VMAX, 'lateral', brain_px), 'pred_med': render_brain(mesh, vals(Pm[t]), VMAX, 'medial', brain_px),
                     'meas_lat': render_brain(mesh, vals(Ym[t]), VMAX, 'lateral', brain_px), 'meas_med': render_brain(mesh, vals(Ym[t]), VMAX, 'medial', brain_px)}
    print('brains rendered', len(brains), flush=True)
    clips = {}
    for e in events:
        sid = e['stimulus_id']
        if not (s0 - 4 <= e['onset_sec'] <= s0 + args.win_s): continue
        p = Path(args.clips) / f'{sid}.mp4'
        frames = []
        if p.exists():
            cap = cv2.VideoCapture(str(p)); nat = cap.get(cv2.CAP_PROP_FPS) or 30
            while True:
                ok, fr = cap.read()
                if not ok: break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
        clips[(e['onset_sec'], sid)] = (frames, nat if frames else 30.0)
    lo, hi = np.percentile(np.stack(list(feat.values())), [2, 98])
    f_small = font(17); f_num = font(26, bold=True); f_mid = font(17)
    n_out = int(round(args.win_s * args.fps)); rows = []
    for j in range(n_out):
        tw = j / args.fps; tabs = s0 + tw; t = min(int(np.floor(tabs / TR)), max(brains)); t = max(t, min(brains))
        canvas = Image.new('RGB', (Wpx, Hpx), PAPER); dr = ImageDraw.Draw(canvas)
        cur = [(o, sid) for (o, sid) in clips if o <= tabs < o + 3.0]
        label = 'blank (1 s between clips)'
        if cur:
            o, sid = cur[0]; frames, nat = clips[(o, sid)]
            if frames:
                fr = Image.fromarray(frames[min(int((tabs - o) * nat), len(frames) - 1)])
                fr = fr.resize((col_w, int(col_w * fr.height / fr.width)))
                if sid in EXCLUDE: fr = fr.filter(ImageFilter.GaussianBlur(18))
                canvas.paste(fr, (x0[0], top + (panel_h - fr.height) // 2))
            label = f'clip {sid.replace("stimulus_", "")} · {tabs - o:0.1f} / 3.0 s' + (' · not shown' if sid in EXCLUDE else '')
            g = feat[sid].reshape(32, 32) if sid in feat else None
            if g is not None: im = Image.fromarray((SEQ_GREY(Normalize(lo, hi)(g)) * 255).astype(np.uint8)[:, :, :3]).resize((col_w, col_w), Image.NEAREST)
            if g is not None: canvas.paste(im, (x0[1], top + (panel_h - col_w) // 2))
            else: dr.text((x0[1], top + panel_h // 2), 'clip outside the model set', fill=MUTED, font=font(17))
        dr.text((x0[0], Hpx - 44), f'scanner time {tabs:0.1f} s · {label}', fill=MUTED, font=f_small)
        dr.text((x0[1], Hpx - 44), 'V-JEPA block 16 · clip-average activity · 1024 units', fill=MUTED, font=f_small)
        for i, key in ((2, 'pred'), (3, 'meas')):
            canvas.paste(brains[t][key + '_lat'], (x0[i], top)); canvas.paste(brains[t][key + '_med'], (x0[i], top + brain_px[1]))
        dr.text((x0[2], Hpx - 44), f'held-out participant · scan {t + 1} · every 1.75 s', fill=MUTED, font=f_small)
        dr.text((x0[3], Hpx - 50), f'match r = {sr[t]:0.2f}', fill=INK, font=f_num)
        dr.text((x0[3] + 190, Hpx - 42), 'participant 1 · single scan · ~5 s lag', fill=MUTED, font=f_mid)
        canvas.save(O / 'frames' / f'f{j:05d}.png'); rows.append([j, round(tw, 3), round(tabs, 3), t, cur[0][1] if cur else '', round(float(sr[t]), 4)])
    with open(O / 'frame_map.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frame', 'wall_time_s', 'scanner_time_s', 'TR', 'clip_id', 'pattern_r_visualROI']); w.writerows(rows)
    json.dump({'run_file': args.run_npz, 'window_scanner_time_s': [s0, s0 + args.win_s], 'TR_s': TR, 'clips_in_window': inside, 'colour_range_z': [-VMAX, VMAX],
               'z_scoring': 'measured BOLD z-scored per run (benchmark convention); predictions on the same scale, no rescaling',
               'vertices_shown': 'visual ROI (4042)', 'views': ['left lateral', 'left medial'], 'selection_rule': 'participant with the best held-out per-TR ROI r (sub-01); 63 s window of their first test run containing the most poster demo clips and none of the display-excluded clips'},
              open(O / 'display_params.json', 'w'), indent=1)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', str(O / 'frames' / 'f%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(O / 'replay_tr.mp4')], check=True)
    print('wrote', n_out, 'frames')

if __name__ == '__main__':
    main()
