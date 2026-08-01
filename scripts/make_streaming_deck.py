"""Build the streaming PowerPoint: a schematic of what the code does, video beneath it.

Two artifacts:
  * schematic.png -- the actual call path, with real class and method names, so a
    technical reader can map every box to a file in the repo
  * streaming_deck.pptx -- title, the main slide (schematic above, video below), the
    depth variant as a backup slide, and a slide of what the demo does NOT claim

The videos are embedded with add_movie, which requires a poster frame; the stills
already rendered serve as posters.

    python unified/scripts/make_streaming_deck.py --out streaming_deck.pptx
"""
import argparse
import os

import numpy as np


def draw_schematic(out_png, dpi=200):
    """The call path, box by box. Every label is a real symbol in the repo."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis('off')

    BLUE, GOLD, GREY, INK = '#2f6bff', '#e0a13b', '#8792a8', '#1b2333'

    def box(x, y, w, h, title, sub, color, fill='#ffffff'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                    linewidth=1.8, edgecolor=color, facecolor=fill))
        ax.text(x + w / 2, y + h * 0.63, title, ha='center', va='center',
                fontsize=11, color=INK, fontweight='bold')
        if sub:
            ax.text(x + w / 2, y + h * 0.26, sub, ha='center', va='center',
                    fontsize=8.5, color=GREY, family='monospace')

    def arrow(x1, y1, x2, y2, color=GREY, style='-|>'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=14, linewidth=1.6, color=color))

    # the world -> a lazy feed
    box(1, 20, 15, 8, 'the clip', '30 s, 30 fps', GREY, '#f5f7fb')
    arrow(16, 24, 21, 24)
    box(21, 20, 17, 8, 'lazy decode', 'decode_frames()\n(a generator)', GREY, '#f5f7fb')
    arrow(38, 24, 43, 24)

    # the interface: this is the part that is the specification, not the model
    box(43, 17, 22, 14, 'WindowedStreamSession', 'window 2000 ms\nstride 500 ms\n'
        'next_input() -> one window', BLUE, '#eef3ff')
    ax.text(54, 32.4, 'the interface', ha='center', fontsize=9.5, color=BLUE,
            style='italic')
    arrow(65, 24, 71, 24, BLUE)

    # the subject
    box(71, 20, 26, 8, 'BrainScoreModel.process(window)', 'start_recording([...taps])',
        BLUE, '#eef3ff')

    # towers
    box(71, 9, 12, 7.5, 'V-JEPA', 'backbone.blocks.N', BLUE, '#eef3ff')
    box(85, 9, 12, 7.5, 'Wav2Vec2', 'encoder.layers.N', GOLD, '#fdf6e8')
    arrow(79, 20, 77, 16.5, BLUE)
    arrow(89, 20, 91, 16.5, GOLD)

    # outputs
    box(71, 0.5, 26, 6.5, 'activations, one vector per window', '', GREY, '#f5f7fb')
    arrow(77, 9, 79, 7, BLUE)
    arrow(91, 9, 89, 7, GOLD)

    # the audio channel joins the same session
    box(21, 6, 17, 8, 'audio feed', 'audio_windows()\nconnects at t = 10 s', GOLD,
        '#fdf6e8')
    arrow(38, 10, 43, 19, GOLD)

    # the one-line claim
    ax.text(50, 34.2, 'One window at a time. Adding a channel adds a feed, not a method.',
            ha='center', fontsize=12.5, color=INK, fontweight='bold')

    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_png


def draw_sequence(out_png, total_s=30.0, connect_s=10.0, stride_s=0.5, dpi=200):
    """How the code RUNS, over time: wire video, run, then wire audio too.

    The static schematic shows what the pieces are. This shows the order they happen
    in, which is the part that carries the claim: the per-window call is identical
    before and after the second channel joins. Only the feed list changes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    BLUE, GOLD, GREY, INK, RED = '#2f6bff', '#e0a13b', '#8792a8', '#1b2333', '#d8483b'
    fig, ax = plt.subplots(figsize=(15, 7.6))
    ax.set_xlim(-6.5, total_s + 1.2); ax.set_ylim(0, 30); ax.axis('off')

    def lane(y, label, color):
        ax.text(-6.2, y, label, ha='left', va='center', fontsize=11,
                color=color, fontweight='bold')
        ax.plot([0, total_s], [y, y], color='#e6eaf2', lw=1.2, zorder=0)

    # time axis
    ax.plot([0, total_s], [2.0, 2.0], color=GREY, lw=1.4)
    for t in range(0, int(total_s) + 1, 5):
        ax.plot([t, t], [1.7, 2.0], color=GREY, lw=1.2)
        ax.text(t, 1.0, f'{t}s', ha='center', fontsize=9, color=GREY)

    # the moment the second channel joins
    ax.plot([connect_s, connect_s], [2.0, 27.5], color=RED, lw=1.5, ls='--', zorder=1)
    ax.text(connect_s + 0.25, 27.8, 'audio connects', fontsize=10, color=RED,
            fontweight='bold')

    ticks = [t for t in np.arange(0, total_s - 2.0 + 1e-9, stride_s)]

    # --- lane 1: video feed --------------------------------------------------
    lane(24.5, 'video feed', BLUE)
    for t in ticks:
        ax.plot([t, t], [24.1, 24.9], color=BLUE, lw=1.1)
    ax.text(connect_s / 2, 26.0, f'one window every {stride_s:g}s  (2 s of frames each)',
            ha='center', fontsize=9.5, color=BLUE)

    # --- lane 2: audio feed --------------------------------------------------
    lane(20.0, 'audio feed', GOLD)
    for t in [t for t in ticks if t >= connect_s]:
        ax.plot([t, t], [19.6, 20.4], color=GOLD, lw=1.1)
    ax.text((connect_s + total_s) / 2, 21.4, 'same cadence, started later',
            ha='center', fontsize=9.5, color=GOLD)

    # --- lane 3: the call ----------------------------------------------------
    lane(14.5, 'the call', INK)
    for t in ticks:
        ax.plot([t, t], [14.1, 14.9], color=INK, lw=0.9, alpha=0.55)
    ax.add_patch(FancyBboxPatch((0.2, 12.0), connect_s - 0.6, 4.6,
                                boxstyle='round,pad=0.25', linewidth=1.6,
                                edgecolor=BLUE, facecolor='#eef3ff', zorder=2))
    ax.text(connect_s / 2, 14.3, 'model.process(window)', ha='center', va='center',
            fontsize=11, family='monospace', color=INK, zorder=3)
    ax.add_patch(FancyBboxPatch((connect_s + 0.4, 12.0), total_s - connect_s - 0.8, 4.6,
                                boxstyle='round,pad=0.25', linewidth=1.6,
                                edgecolor=GOLD, facecolor='#fdf6e8', zorder=2))
    ax.text((connect_s + total_s) / 2, 14.3, 'model.process(window)', ha='center',
            va='center', fontsize=11, family='monospace', color=INK, zorder=3)
    ax.text((connect_s + total_s) / 2, 12.6, 'identical call', ha='center',
            fontsize=9, color=GOLD, style='italic', zorder=3)

    # --- lane 4: what comes back --------------------------------------------
    lane(8.0, 'what comes back', GREY)
    ax.add_patch(FancyBboxPatch((0.2, 6.2), connect_s - 0.6, 3.4,
                                boxstyle='round,pad=0.2', linewidth=1.4,
                                edgecolor=BLUE, facecolor='#eef3ff'))
    ax.text(connect_s / 2, 7.9, 'video units', ha='center', va='center',
            fontsize=10, color=INK)
    ax.add_patch(FancyBboxPatch((connect_s + 0.4, 6.2), total_s - connect_s - 0.8, 3.4,
                                boxstyle='round,pad=0.2', linewidth=1.4,
                                edgecolor=GOLD, facecolor='#fdf6e8'))
    ax.text((connect_s + total_s) / 2, 7.9, 'video units  +  audio units',
            ha='center', va='center', fontsize=10, color=INK)

    # --- the two setup lines, at the moments they run ------------------------
    ax.annotate("session = WindowedStreamSession(video_feed, window_ms=2000, stride_ms=500)\n"
                "model.start_recording('video_mid')",
                xy=(0, 27.0), xytext=(0, 28.4), fontsize=9.5, family='monospace',
                color=BLUE, ha='left', va='bottom')
    ax.annotate("model.start_recording(['video_mid', 'audio_mid'])",
                xy=(connect_s, 22.6), xytext=(connect_s + 0.3, 23.2), fontsize=9.5,
                family='monospace', color=GOLD, ha='left', va='bottom')

    ax.text(total_s / 2, 0.0,
            'Wiring a second channel changes the recording list, not the loop.',
            ha='center', fontsize=12, color=INK, fontweight='bold')

    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_png


def build_deck(out_pptx, schematic, sequence, channels_mp4, channels_poster,
               depths_mp4, depths_poster):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    BLANK = prs.slide_layouts[6]
    INK = RGBColor(0x1b, 0x23, 0x33)
    GREY = RGBColor(0x87, 0x92, 0xa8)

    def textbox(slide, x, y, w, h, text, size=18, bold=False, color=INK):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = text
        p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
        return tf

    # --- title ---------------------------------------------------------------
    s = prs.slides.add_slide(BLANK)
    textbox(s, 0.9, 2.4, 11.5, 1.2, 'Streaming, in the interface', 40, True)
    textbox(s, 0.9, 3.6, 11.5, 1.6,
            'A continuous stream arrives, the interface hands the model one window at '
            'a time, and a second channel can join mid-stream without changing the call.',
            18, False, GREY)

    # --- main slide: schematic above, video below ----------------------------
    s = prs.slides.add_slide(BLANK)
    textbox(s, 0.5, 0.22, 12.3, 0.5, 'What the code does', 24, True)
    s.shapes.add_picture(schematic, Inches(0.5), Inches(0.8), width=Inches(12.3))
    if os.path.exists(channels_mp4):
        s.shapes.add_movie(channels_mp4, Inches(2.1), Inches(3.55), Inches(9.1),
                           Inches(3.5), poster_frame_image=channels_poster,
                           mime_type='video/mp4')
    textbox(s, 0.5, 7.05, 12.3, 0.4,
            'Video tower V-JEPA v1, audio tower Wav2Vec2-base; 2000 ms windows, '
            '500 ms stride; audio connects at t = 10 s.', 11, False, GREY)

    # --- how it runs, over time ---------------------------------------------
    s = prs.slides.add_slide(BLANK)
    textbox(s, 0.5, 0.22, 12.3, 0.5, 'How it runs', 24, True)
    s.shapes.add_picture(sequence, Inches(0.6), Inches(0.85), width=Inches(12.1))
    textbox(s, 0.5, 6.9, 12.3, 0.5,
            'The per-window call is the same before and after the second channel '
            'joins; only the recording list changes.', 12, False, GREY)

    # --- backup: depth variant ----------------------------------------------
    s = prs.slides.add_slide(BLANK)
    textbox(s, 0.5, 0.22, 12.3, 0.5, 'Backup: reading three depths at once', 24, True)
    if os.path.exists(depths_mp4):
        s.shapes.add_movie(depths_mp4, Inches(2.4), Inches(0.95), Inches(8.5),
                           Inches(5.1), poster_frame_image=depths_poster,
                           mime_type='video/mp4')
    textbox(s, 0.5, 6.2, 12.3, 1.1,
            'Three taps on the video tower, one shared forward pass. Measured '
            'adjacent-window similarity: 0.94 early, 0.85 middle, 0.79 late — deeper '
            'layers change fastest, which is the opposite of what we expected before '
            'looking.', 13, False, GREY)

    # --- what this does not claim -------------------------------------------
    s = prs.slides.add_slide(BLANK)
    textbox(s, 0.7, 0.5, 12.0, 0.6, 'What this does not claim', 26, True)
    for i, line in enumerate([
        'Scoring still runs in batch. Batched delivery is the default on the '
        'perception and behavior paths, deliberately, for throughput and to reproduce '
        'existing scores exactly.',
        'The movie benchmarks are not streaming. They build a stimulus set and score '
        'it; this demo is the streaming path running alongside, not those benchmarks '
        'rewritten.',
        'No brain data is involved. The strips are the model\'s own units. Nothing '
        'here is a claim about cortex.',
        'Real-time is a separate step. Pacing against a wall clock, with an explicit '
        'policy for falling behind, is implemented but not shown here.',
    ]):
        textbox(s, 0.9, 1.5 + i * 1.32, 11.6, 1.2, '— ' + line, 15, False, INK)

    prs.save(out_pptx)
    return out_pptx


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, 'streaming_demo_output')
    ap.add_argument('--out', default=os.path.join(d, 'streaming_deck.pptx'))
    ap.add_argument('--dir', default=d)
    args = ap.parse_args()

    schematic = draw_schematic(os.path.join(args.dir, 'schematic.png'))
    print('schematic ->', schematic, flush=True)
    sequence = draw_sequence(os.path.join(args.dir, 'sequence.png'))
    print('sequence ->', sequence, flush=True)
    out = build_deck(
        args.out, schematic, sequence,
        os.path.join(args.dir, 'streaming_channels.mp4'),
        os.path.join(args.dir, 'poster_channels.png'),
        os.path.join(args.dir, 'streaming_depths.mp4'),
        os.path.join(args.dir, 'poster_depths.png'))
    print('deck ->', out, flush=True)


if __name__ == '__main__':
    main()
