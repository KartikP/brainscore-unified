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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--replay', required=True)
    ap.add_argument('--poster_frame', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--algonauts', default=None)
    ap.add_argument('--table_json', required=True)
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
    heads = [('Watch', 'A 3-second clip the participants saw'),
             ('Record', 'Activity inside the model, patch by patch'),
             ('Predict', 'A fitted map turns that activity into a brain response'),
             ('Compare', 'The response recorded in 10 people')]
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
                    f'ticks = participant range (Friends, Courtois NeuroMod).')
        else:
            rb = algo['subjects'][subs[0]]['run_bootstrap']
            def tick(c): return tuple(rb[c]['median_r_ci95'])
            unit = ('Held-out correlation, median over 1000 brain parcels, one participant; '
                    'ticks = 95% interval over runs (Friends, Courtois NeuroMod).')
        add_text(s, 'Section 3 caption', x, by + 0.34, sec_w, 0.4,
                 'Same method, a TV show watched with sound: sound and words each add to what video predicts. '
                 'That tells us which brain experiment to run next.', 10.5, MUTED)
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
    prs.save(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
