"""Build the single-slide animated donor poster (20 x 11.25 in, 16:9 TV).

Everything except the measured replay is native, editable PowerPoint: text,
the green model table, the two modality panels (rectangles + text), notes.
The replay is an embedded MP4 that autoplays and loops (timing XML injected).

Usage:
  python build_poster.py --data <visualization-data dir> --replay replay.mp4 --out poster.pptx
"""
import argparse
import copy
import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

FONT = 'Helvetica Neue'
INK = RGBColor(0x14, 0x14, 0x14)
MUTED = RGBColor(0x6E, 0x6E, 0x69)
PAPER = RGBColor(0xF6, 0xF5, 0xF1)
RULE = RGBColor(0xD7, 0xD6, 0xD0)
BLUE = RGBColor(0x2A, 0x78, 0xD6)     # video
ORANGE = RGBColor(0xEB, 0x68, 0x34)   # sound
AQUA = RGBColor(0x1B, 0xAF, 0x7A)     # words
GREY_BAR = RGBColor(0x9A, 0x99, 0x93)
# green ordinal ramp (light end >= 2:1 on paper), validated
GREENS = ['#bfe3cb', '#95cfa9', '#6ab987', '#44a068', '#2a844f', '#1a673b']

SLIDE_W, SLIDE_H = 20.0, 11.25
# replay canvas geometry must match render_replay.py
REPLAY_PX = (1680, 480)
PAD_PX, COL_PX = 24, (1680 - 5 * 24) // 4
REPLAY_BOX = (0.8, 1.98, 18.4, 18.4 * REPLAY_PX[1] / REPLAY_PX[0])   # x, y, w, h inches



def _paired_or_synth(A):
    """Use the cross-participant block if the run finished, else synthesize it from the per-participant entries."""
    if A.get('paired_across_subjects'):
        return A['paired_across_subjects']
    subs = list(A['subjects'])
    conds = [c for c in ('video', 'audio', 'text', 'video+audio', 'video+audio+text', 'concat') if all(c in A['subjects'][s] for s in subs)]
    return {'subjects': subs, 'per_subject_median_r': {c: [A['subjects'][s][c]['median_r'] for s in subs] for c in conds}, 'partial': True}

def rgb(hexstr):
    h = hexstr.lstrip('#')
    return RGBColor(int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


def add_text(slide, name, x, y, w, h, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split('\n') if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def add_rect(slide, name, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(0.75)
    return s


def inject_autoplay_loop(slide, movie_shape):
    """Timing tree: play the media automatically on slide start, loop until stopped."""
    P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
    spid = movie_shape.shape_id
    xml = f'''
<p:timing xmlns:p="{P}">
 <p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>
  <p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>
   <p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>
    <p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
     <p:par><p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="withEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
      <p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="6" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:cmd>
     </p:childTnLst></p:cTn></p:par>
    </p:childTnLst></p:cTn></p:par>
   </p:childTnLst></p:cTn></p:par>
  </p:childTnLst></p:cTn>
  <p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>
  <p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>
  </p:seq>
  <p:video><p:cMediaNode vol="80000" showWhenStopped="0"><p:cTn id="7" repeatCount="indefinite" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cMediaNode></p:video>
 </p:childTnLst></p:cTn></p:par></p:tnLst>
</p:timing>'''
    timing = etree.fromstring(xml)
    sld = slide._element
    for old in sld.findall(f'{{{P}}}timing'):
        sld.remove(old)
    # timing must come after clrMapOvr / transition and before extLst
    ext = sld.find(f'{{{P}}}extLst')
    if ext is not None:
        ext.addprevious(timing)
    else:
        sld.append(timing)


def bar_panel(slide, name, x, y, w, h, rows, vmax, unit_note):
    """rows: list of (label, value, (lo, hi) or None, color). Horizontal thin bars with CI ticks."""
    label_w = 1.75
    bar_x = x + label_w
    bar_w = w - label_w - 0.75
    row_h = h / len(rows)
    # baseline
    add_rect(slide, f'{name} baseline', bar_x, y, 0.012, h, RULE)
    for i, (lab, val, ci, col) in enumerate(rows):
        cy = y + i * row_h + row_h * 0.5
        add_text(slide, f'{name} label {i}', x, cy - 0.17, label_w - 0.1, 0.34, lab, 14, INK, anchor=MSO_ANCHOR.MIDDLE)
        bw = max(0.02, bar_w * val / vmax)
        add_rect(slide, f'{name} bar {i}', bar_x, cy - 0.11, bw, 0.22, col, shape=MSO_SHAPE.RECTANGLE)
        if ci is not None:
            lo, hi = ci
            add_rect(slide, f'{name} ci {i}', bar_x + bar_w * lo / vmax, cy - 0.005, max(0.01, bar_w * (hi - lo) / vmax), 0.01, INK)
            add_rect(slide, f'{name} ci lo {i}', bar_x + bar_w * lo / vmax, cy - 0.055, 0.012, 0.11, INK)
            add_rect(slide, f'{name} ci hi {i}', bar_x + bar_w * hi / vmax, cy - 0.055, 0.012, 0.11, INK)
        right_edge = bar_x + max(bw, bar_w * (ci[1] / vmax) if ci else 0)
        add_text(slide, f'{name} value {i}', right_edge + 0.1, cy - 0.17, 0.7, 0.34, f'{val:0.2f}'.replace('-0.00', '0.00'), 14, INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, f'{name} unit', bar_x, y + h + 0.02, bar_w + 0.7, 0.25, unit_note, 10, MUTED)


SESS = ['baseline', 'lesion_pseudo_selective', 'control_random', 'lesion_real_selective', 'restored']
SESS_LABEL = ['before', 'targeted', 'random', 'opposite', 'restored']
PROTOCOL = """subjects = 3 model sizes x 7 decoding settings     # 21 agents
words    = 100 held-out word images                 # real + made-up

for each subject:
    show(words); record(answers, activity)          # 1 before
    localize(400 training words)                    # units that prefer made-up words
    silence(those units: 25% of 5 late layers)
    show(words); record(answers, activity)          # 2 targeted lesion
    silence(random units, same number); show(words) # 3 control
    silence(units that prefer real words); show(words)  # 4 opposite
    reset(); show(words); record(answers, activity) # 5 restored"""


def _size_label(tag):
    import re
    m = re.search(r'(\d+)b', tag)
    return f'{m.group(1)} billion' if m else tag


def _mono_text(slide, name, x, y, w, h, text, size):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.18); tf.margin_top = tf.margin_bottom = Inches(0.12)
    first = True
    for line in text.split('\n'):
        para = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        code, _, comment = line.partition('#')
        r = para.add_run(); r.text = code.rstrip() if comment else line
        r.font.name = 'Menlo'; r.font.size = Pt(size); r.font.color.rgb = INK
        if comment:
            r2 = para.add_run(); r2.text = '   # ' + comment.strip()
            r2.font.name = 'Menlo'; r2.font.size = Pt(size); r2.font.color.rgb = MUTED
    return tb


def build_slide2(prs, args):
    S = json.load(open(args.lesion_summary))
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = PAPER
    add_text(s, 'Brand', 0.8, 0.36, 6, 0.4, 'Brain-Score', 16, INK, bold=True)
    add_text(s, 'Affiliation', 11.2, 0.36, 8.0, 0.4, 'MIT Siegel Family Quest for Intelligence', 13, MUTED, align=PP_ALIGN.RIGHT)
    add_text(s, 'Headline', 0.8, 0.72, 16, 0.75, 'Models as experimental subjects', 40, INK, bold=True)
    add_text(s, 'Purpose', 0.8, 1.36, 17.5, 0.4,
             'A lesion study that could never be done in a person takes minutes in a model, can be repeated across a whole population of them, and can be undone.',
             17, MUTED)
    add_rect(s, 'Rule', 0.8, 1.9, 18.4, 0.012, RULE)
    # ---- left: protocol ----
    x = 0.8; w = 7.0
    add_text(s, 'Protocol heading', x, 2.05, w, 0.4, '1.  The protocol, written as the lab runs it', 19, INK, bold=True)
    add_rect(s, 'Protocol box', x, 2.55, w, 2.75, RGBColor(0xEE, 0xED, 0xE8))
    _mono_text(s, 'Protocol code', x, 2.55, w, 2.75, PROTOCOL, 10.5)
    add_text(s, 'Protocol caption', x, 5.45, w, 1.3,
             'Each line is one call to the same toolbox used on the first slide: show stimuli and record (process), change the model\'s state '
             '(process with a StateChange), and reset. The word images and the pass mark come from a published reading test (Yeatman 2021); '
             'the localizer is the classic neuroscience move of finding units by what they prefer before touching them.', 11, MUTED)

    # overall accuracy table (mean over agents), under the protocol
    ty = 6.85
    add_text(s, 'Table heading', x, ty, w, 0.3, 'All 100 words, share correct (mean over agents)', 11, INK, bold=True)
    models_sorted = sorted(S['models'], key=lambda m: S['models'][m]['hidden'])
    shape = s.shapes.add_table(len(models_sorted) + 1, 6, Inches(x), Inches(ty + 0.35), Inches(w), Inches(0.27 * (len(models_sorted) + 1)))
    shape.name = 'Overall accuracy table'
    tbl = shape.table
    tbl.columns[0].width = Inches(1.6)
    for c in range(1, 6): tbl.columns[c].width = Inches((w - 1.6) / 5)
    heads = ['model size'] + SESS_LABEL
    for c, hd in enumerate(heads):
        cell = tbl.cell(0, c); cell.text = hd
    for r_, m in enumerate(models_sorted, start=1):
        tbl.cell(r_, 0).text = _size_label(m)
        for c, sess in enumerate(SESS, start=1):
            vals = [a['sessions'][sess]['accuracy'] for a in S['models'][m]['agents'].values()]
            tbl.cell(r_, c).text = f'{sum(vals) / len(vals):0.2f}'
    for r_ in range(len(models_sorted) + 1):
        tbl.rows[r_].height = Inches(0.27)
        for c in range(6):
            cell = tbl.cell(r_, c); cell.fill.solid(); cell.fill.fore_color.rgb = PAPER if r_ else RGBColor(0xEE, 0xED, 0xE8)
            cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.02)
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                for run in para.runs:
                    run.font.size = Pt(10); run.font.name = FONT; run.font.color.rgb = INK if r_ else MUTED; run.font.bold = (c == 0 and r_ > 0)
    add_text(s, 'Table note', x, ty + 0.35 + 0.27 * (len(models_sorted) + 1) + 0.05, w, 0.3,
             'Pass mark in the human reading test: 0.65. Human mean on these 100 words: 0.81.', 9.5, MUTED)
    # across sizes, computed from the data
    rows_txt = []
    for m in models_sorted:
        pop = S['models'][m]['population']; tm = pop['targeted_minus_random_deficit']['per_agent']
        ag = S['models'][m]['agents']
        dp = [a['sessions']['lesion_pseudo_selective']['deficit_pseudo'] for a in ag.values()]
        dr = [a['sessions']['control_random']['deficit_pseudo'] for a in ag.values()]
        rows_txt.append(f"{_size_label(m)}: made-up-word drop {sum(dp)/len(dp):+0.2f} targeted vs {sum(dr)/len(dr):+0.2f} random (agent range {min(dp):+0.2f} to {max(dp):+0.2f} targeted)")
    if 'dose' in S:   # lesion-size sweep, greedy agent: lowest made-up-word accuracy over all fractions tried
        for m in models_sorted:
            if m not in S['dose']:
                continue
            fr = S['dose'][m]; fmax = max(float(f) for f in fr)
            lo_t = min(d['lesion_pseudo_selective'] for d in fr.values()); lo_r = min(d['control_random'] for d in fr.values())
            rows_txt.append(f"{_size_label(m)}, up to {fmax:.0%} of units silenced: made-up-word accuracy never below {lo_t:.2f} targeted, {lo_r:.2f} random")
    add_text(s, 'Across sizes heading', x, ty + 0.35 + 0.27 * (len(models_sorted) + 1) + 0.45, w, 0.3, 'Across sizes', 11, INK, bold=True)
    add_text(s, 'Across sizes text', x, ty + 0.35 + 0.27 * (len(models_sorted) + 1) + 0.78, w, 1.6, '\n'.join(rows_txt), 10, MUTED)
    # ---- middle: results chart (made-up word accuracy per session, per size; dots = agents) ----
    models = sorted(S['models'], key=lambda m: S['models'][m]['hidden'])
    cx = 8.2; cw = 6.1
    add_text(s, 'Results heading', cx, 2.05, cw, 0.4, '2.  What happened to reading made-up words', 19, INK, bold=True)
    add_text(s, 'Results caption', cx, 2.45, cw, 0.55,
             'Share of made-up words answered correctly, before and after each kind of lesion. Bars: mean over agents; dots: each agent (temperatures 0, 0.5, 1).', 10.5, MUTED)
    gx = cx + 1.25; gw = cw - 1.25; col_w = gw / 5; row_h = 2.0; top = 3.6; bar_h = 1.45
    for j, lab in enumerate(SESS_LABEL):
        add_text(s, f'Col head {j}', gx + j * col_w, top - 0.5, col_w, 0.3, lab, 10, MUTED, align=PP_ALIGN.CENTER)
    for i, m in enumerate(models):
        ry = top + i * row_h; base_y = ry + bar_h
        add_text(s, f'Row label {i}', cx, ry + bar_h - 0.5, 1.2, 0.5, f'{_size_label(m)}\nparameters', 11, INK, bold=True, anchor=MSO_ANCHOR.BOTTOM)
        add_rect(s, f'Row baseline {i}', gx, base_y, gw, 0.012, RULE)
        add_rect(s, f'Row half {i}', gx, base_y - 0.5 * bar_h, gw, 0.006, RGBColor(0xE2, 0xE1, 0xDB))
        agents = S['models'][m]['agents']
        for j, sess in enumerate(SESS):
            vals = [a['sessions'][sess]['accuracy_pseudo'] for a in agents.values() if a['sessions'][sess].get('accuracy_pseudo') is not None]
            mean = sum(vals) / len(vals)
            bw = col_w * 0.46; bx = gx + j * col_w + (col_w - bw) / 2
            col = BLUE if sess == 'baseline' else (RGBColor(0xC9, 0x3C, 0x3C) if sess == 'lesion_pseudo_selective' else GREY_BAR)
            add_rect(s, f'Bar {i} {j}', bx, base_y - bar_h * mean, bw, max(0.015, bar_h * mean), col)
            for k, v in enumerate(vals):
                d = 0.085; dx = bx + bw * (0.15 + 0.7 * (k / max(1, len(vals) - 1)))
                add_rect(s, f'Dot {i} {j} {k}', dx - d / 2, base_y - bar_h * v - d / 2, d, d, INK, shape=MSO_SHAPE.OVAL)
            add_text(s, f'Val {i} {j}', gx + j * col_w, base_y - bar_h * max(vals + [mean]) - 0.3, col_w, 0.25, f'{mean:0.2f}', 10, INK, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, 'Chart unit', cx, top + len(models) * row_h + 0.02, cw, 0.3,
             'Thin line = coin flip (0.5). Real-word answers stayed near ceiling in every session (notes).', 9.5, MUTED)
    # ---- right: recording + population ----
    rx = 14.7; rw = 4.5
    add_text(s, 'Recording heading', rx, 2.05, rw, 0.4, '3.  The recording shows the lesion', 19, INK, bold=True)
    if args.raster_png and Path(args.raster_png).exists():
        pic = s.shapes.add_picture(args.raster_png, Inches(rx), Inches(2.55), width=Inches(rw)); pic.name = 'Activity raster'
        pic_bottom = 2.55 + Emu(pic.height).inches
    else:
        pic_bottom = 2.6
    add_text(s, 'Recording caption', rx, pic_bottom + 0.08, rw, 0.9,
             'Activity of one late layer on every trial (one column per word image, one row per unit, silenced units at the top): '
             'before, and during the targeted lesion. The same recording hooks that measured the model on slide one.', 10.5, MUTED)
    # population text, computed
    lines = []
    for m in models:
        pop = S['models'][m]['population']; n = pop['baseline']['n_agents']
        tm = pop['targeted_minus_random_deficit']
        lines.append(f"{_size_label(m)}: targeted lesion cost more than random in {tm['n_agents_targeted_worse_than_random']} of {n} agents, "
                     f"beyond trial-to-trial noise in {tm['n_agents_reliable']}; "
                     f"after reset {'all ' + str(n) if pop['restored_identical_all_agents'] else 'not all'} agents gave their original answers.")
    add_text(s, 'Population heading', rx, pic_bottom + 1.05, rw, 0.35, 'A population, not a single subject', 14, INK, bold=True)
    add_text(s, 'Population text', rx, pic_bottom + 1.42, rw, 1.8, '\n'.join(lines), 10.5, MUTED)
    add_text(s, 'Website', 0.8, SLIDE_H - 0.36, 10, 0.3, 'brain-score.org', 11, MUTED)
    add_text(s, 'Payoff', 9.0, SLIDE_H - 0.36, 10.2, 0.3,
             'Nothing on this slide could be done to a person. In a model it took minutes, and every subject was restored.', 11, INK, align=PP_ALIGN.RIGHT)
    notes = s.notes_slide.notes_text_frame
    notes.text = open(args.lesion_notes).read() if args.lesion_notes and Path(args.lesion_notes).exists() else 'see methods document'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--replay', required=True)
    ap.add_argument('--poster_frame', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--algonauts', default=None)
    ap.add_argument('--table_json', required=True)
    ap.add_argument('--lesion_summary', default=None)
    ap.add_argument('--raster_png', default=None)
    ap.add_argument('--lesion_notes', default=None)
    args = ap.parse_args()
    D = Path(args.data)
    R = json.load(open(D / 'metrics' / 'results_main.json'))
    boot = R.get('clip_bootstrap_subject_mean', {}).get('visual', {})
    null = R.get('permutation_null', {}).get('visual', {})
    def ci(c):
        v = boot.get(c, {}).get('median_r_ci95'); return tuple(v) if v else None
    mean = R['target_subject_mean']
    algo = json.load(open(args.algonauts)) if args.algonauts and Path(args.algonauts).exists() else None
    table = json.load(open(args.table_json))

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W); prs.slide_height = Inches(SLIDE_H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = PAPER

    # ---- header (compact) ----
    add_text(s, 'Brand', 0.8, 0.36, 6, 0.4, 'Brain-Score', 16, INK, bold=True)
    add_text(s, 'Affiliation', 11.2, 0.36, 8.0, 0.4, 'MIT Siegel Family Quest for Intelligence', 13, MUTED, align=PP_ALIGN.RIGHT)
    add_text(s, 'Headline', 0.8, 0.72, 14, 0.75, 'The Neuroscientific Toolbox for AI', 40, INK, bold=True)
    add_text(s, 'Purpose', 0.8, 1.36, 17.5, 0.4,
             'The brain is hard to study directly. AI models are imperfect stand-ins we can open, measure, and test against real recordings.',
             17, MUTED)

    # ---- replay: native column headings above the embedded video ----
    rx, ry, rw, rh = REPLAY_BOX
    scale = rw / REPLAY_PX[0]
    heads = [('Watch', 'What participants watched: silent 3 s clips, then a TV episode with sound'),
             ('Record', 'Activity inside the model as it watches'),
             ('Predict', 'A fitted map turns that activity into a brain response'),
             ('Compare', 'The response recorded in people')]
    for i, (hd, sb) in enumerate(heads):
        cx = rx + (PAD_PX + i * (COL_PX + PAD_PX)) * scale
        cw = COL_PX * scale
        add_text(s, f'Stage {i} heading', cx, ry, cw, 0.38, f'{i + 1}.  {hd}', 19, INK, bold=True)
        add_text(s, f'Stage {i} caption', cx, ry + 0.34, cw, 0.3, sb, 11, MUTED)
    video_y = ry + 0.72
    mv = s.shapes.add_movie(args.replay, Inches(rx), Inches(video_y), Inches(rw), Inches(rh),
                            poster_frame_image=args.poster_frame, mime_type='video/mp4')
    mv.name = 'Measured replay (embedded video, autoplay + loop)'
    inject_autoplay_loop(s, mv)

    # ---- bottom band: three sections ----
    by = video_y + rh + 0.4
    add_rect(s, 'Rule', 0.8, by - 0.2, 18.4, 0.012, RULE)
    sec_w = 5.75
    gaps = [0.8, 0.8 + sec_w + 0.575, 0.8 + 2 * (sec_w + 0.575)]

    # 1. find promising models — green table (editable native table)
    x = gaps[0]
    add_text(s, 'Section 1 heading', x, by, sec_w, 0.35, 'Find promising models', 17, INK, bold=True)
    add_text(s, 'Section 1 caption', x, by + 0.34, sec_w, 0.4,
             'How well each model predicts visual-cortex responses to these clips, relative to how well people agree with each other.', 10.5, MUTED)
    rows = sorted(table['rows'], key=lambda r: -r['normalized'])
    names = {'vjepa1-vitl': 'V-JEPA (video)', 'clip-vit-b-32': 'CLIP (image + text)', 'vjepa2-vitl': 'V-JEPA 2 (video)',
             'videomae-base': 'VideoMAE (video)', 'qwen2.5-vl-3b': 'Qwen2.5-VL (vision + language)', 'blip2-opt-2.7b': 'BLIP-2 (vision + language)'}
    tbl_y = by + 0.82
    shape = s.shapes.add_table(len(rows), 2, Inches(x), Inches(tbl_y), Inches(sec_w), Inches(0.235 * len(rows)))
    shape.name = 'Model table'
    tbl = shape.table
    tbl.columns[0].width = Inches(sec_w - 1.1); tbl.columns[1].width = Inches(1.1)
    tblPr = tbl._tbl.tblPr
    tblPr.set('firstRow', '0'); tblPr.set('bandRow', '0')
    for i, r in enumerate(rows):
        tbl.rows[i].height = Inches(0.235)
        c0, c1 = tbl.cell(i, 0), tbl.cell(i, 1)
        c0.text = names.get(r['model'], r['model']); c1.text = f"{r['normalized']:0.2f}"
        for c, al in ((c0, PP_ALIGN.LEFT), (c1, PP_ALIGN.CENTER)):
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]; p.alignment = al
            p.runs[0].font.size = Pt(11); p.runs[0].font.name = FONT; p.runs[0].font.color.rgb = INK
            p.runs[0].font.bold = (c is c1)
        c0.fill.solid(); c0.fill.fore_color.rgb = PAPER
        gi = min(len(GREENS) - 1, int(r['normalized'] / (1.0 / len(GREENS))))
        c1.fill.solid(); c1.fill.fore_color.rgb = rgb(GREENS[gi])
    add_text(s, 'Section 1 unit', x, tbl_y + 0.235 * len(rows) + 0.05, sec_w, 0.25,
             'Ceiling-normalized prediction correlation, visual cortex, BOLDMoments (Lahner 2024).', 9.5, MUTED)

    # 2. investigate — which inputs carry the signal (BOLDMoments, silent viewing)
    x = gaps[1]
    add_text(s, 'Section 2 heading', x, by, sec_w, 0.35, 'Investigate what happens inside', 17, INK, bold=True)
    add_text(s, 'Section 2 caption', x, by + 0.34, sec_w, 0.4,
             'Which inputs carry the signal? These clips were watched in silence: the soundtrack alone predicts a little, '
             'adding it to video adds nothing, and the method says so.', 10.5, MUTED)
    v = mean['video']['roi']['visual']['median_r']; a = mean['audio']['roi']['visual']['median_r']; b = mean['both']['roi']['visual']['median_r']
    rows2 = [('Video', v, ci('video'), BLUE),
             ('Soundtrack', a, ci('audio'), ORANGE),
             ('Video + sound', b, ci('both'), GREY_BAR),
             ('Shuffled clips', null.get('video', {}).get('null_mean', 0.0), None, GREY_BAR)]
    bar_panel(s, 'Panel 2', x, by + 0.86, sec_w, 1.24, rows2, 0.7,
              'Held-out correlation, median over 4042 visual-cortex vertices, 10 participants averaged; ticks = 95% interval over clips.')

    # 3. test & predict — Friends with sound (if available)
    x = gaps[2]
    add_text(s, 'Section 3 heading', x, by, sec_w, 0.35, 'Test explanations, make predictions', 17, INK, bold=True)
    if algo and algo.get('subjects'):
        PA = _paired_or_synth(algo)
        ps = PA['per_subject_median_r']
        subs = PA['subjects']
        import statistics as st
        def m(c): return st.mean(ps[c])
        if len(subs) >= 2:
            def tick(c): return (min(ps[c]), max(ps[c]))
            unit = (f'Held-out correlation, median over 1000 brain parcels; mean of {len(subs)} participants, '
                    f'ticks = participant range (Friends).')
        else:
            rb = algo['subjects'][subs[0]]['run_bootstrap']
            def tick(c): return tuple(rb[c]['median_r_ci95'])
            unit = ('Held-out correlation, median over 1000 brain parcels, one participant; '
                    'ticks = 95% interval over runs (Friends, Courtois NeuroMod).')
        add_text(s, 'Section 3 caption', x, by + 0.34, sec_w, 0.4,
                 'Same method, a TV show watched with sound: sound and words each add to what video predicts. '
                 'The replay above ends with one of these episodes, scan by scan.', 10.5, MUTED)
        rows3 = [('Video', m('video'), tick('video'), BLUE), ('Sound', m('audio'), tick('audio'), ORANGE),
                 ('Words', m('text'), tick('text'), AQUA), ('All three', m('video+audio+text'), tick('video+audio+text'), GREY_BAR)]
        bar_panel(s, 'Panel 3', x, by + 0.86, sec_w, 1.24, rows3, 0.35, unit)
    else:
        add_text(s, 'Section 3 caption', x, by + 0.34, sec_w, 1.4,
                 'The same measurements can be repeated with parts of the model switched off, or with a sense withheld, '
                 'to test which features carry the signal and to choose the next brain experiment.', 11, MUTED)

    add_text(s, 'Website', 0.8, SLIDE_H - 0.36, 10, 0.3, 'brain-score.org', 11, MUTED)
    add_text(s, 'Payoff', 9.0, SLIDE_H - 0.36, 10.2, 0.3,
             'Every prediction shown here was checked against human recordings the model never saw.', 11, INK, align=PP_ALIGN.RIGHT)

    # ---- notes: provenance + limits ----
    notes = s.notes_slide.notes_text_frame
    notes.text = open(D / 'poster_notes.txt').read() if (D / 'poster_notes.txt').exists() else 'see methods document'
    if args.lesion_summary:
        build_slide2(prs, args)
    prs.save(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
