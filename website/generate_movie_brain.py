"""Generate frames for the movie+audio+text -> evolving fMRI BOLD panel.

An ILLUSTRATIVE 11-second clip: per second it renders (a) a movie frame, and
(b) a predicted cortical BOLD map that EVOLVES as the clip plays — visual cortex
(occipital, back) tracks the picture, auditory cortex (temporal) tracks the
sound, and the language network (frontal/temporal) lights when words are spoken.
A waveform strip + per-second transcript complete the multimodal input.

This is a *concept* animation of the temporal-multimodal -> BOLD pipeline (per-TR
features -> HRF -> per-parcel ridge prediction). The activation pattern here is
synthesised to mirror that structure; the real per-TR predictions run on EC2.
Fast matplotlib only — no model, no GPU.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = os.path.join(os.path.dirname(__file__), 'assets', 'movie_brain')
N = 11                      # seconds (>= 10s clip), 1 frame/sec
W, H = 200, 180             # cortex grid

# per-second transcript (when each word is "spoken")
TRANSCRIPT = ['a', 'quiet', 'street', 'at', 'dusk', '—', 'a car', 'passes',
              'by', 'and', 'fades']
SPEECH = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=float)  # speech present?


# --- a lateral cortex silhouette (closed outline), in grid coords ---
def cortex_outline():
    # hand-tuned lateral-view brain blob (x right→front, y up)
    t = np.linspace(0, 2 * np.pi, 240)
    cx, cy, rx, ry = 0.50, 0.50, 0.42, 0.30
    x = cx + rx * np.cos(t) * (1 + 0.10 * np.cos(2 * t) + 0.06 * np.cos(3 * t))
    y = cy + ry * np.sin(t) * (1 + 0.12 * np.cos(t))
    # flatten the bottom (temporal lobe sits lowish), add a frontal bulge
    y = y - 0.05 * np.exp(-((x - 0.78) ** 2) / 0.02)
    return np.column_stack([x * W, y * H]), (cx, cy, rx, ry)


def gaussian(gx, gy, cx, cy, sx, sy):
    return np.exp(-(((gx - cx) ** 2) / (2 * sx ** 2) + ((gy - cy) ** 2) / (2 * sy ** 2)))


def bold_frame(i, outline, params):
    """Predicted BOLD over cortex at second i, as evolving Gaussian sources."""
    cx, cy, rx, ry = params
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    gx, gy = np.meshgrid(xs, ys)
    tfrac = i / (N - 1)
    # source intensities (the multimodal drivers), smoothly varying in time
    visual = 0.55 + 0.45 * np.sin(2 * np.pi * (tfrac * 1.5))        # picture energy
    audio = 0.45 + 0.45 * np.sin(2 * np.pi * (tfrac * 1.0) + 1.0)   # sound envelope
    lang = SPEECH[i] * (0.5 + 0.5 * np.sin(2 * np.pi * tfrac * 2))  # speech-driven
    z = (
        visual * gaussian(gx, gy, 0.18, 0.50, 0.07, 0.10) +        # occipital (back)
        0.6 * visual * gaussian(gx, gy, 0.30, 0.62, 0.06, 0.07) +  # higher visual
        audio * gaussian(gx, gy, 0.52, 0.34, 0.07, 0.06) +         # auditory (temporal)
        lang * gaussian(gx, gy, 0.72, 0.46, 0.07, 0.09)            # language (frontal/temporal)
    )
    # mask to the cortex ellipse
    mask = (((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2) <= 1.0
    z = np.where(mask, z, np.nan)
    return z


def render_bold(i, outline, params):
    z = bold_frame(i, outline, params)
    cx, cy, rx, ry = params
    fig, ax = plt.subplots(figsize=(3.2, 2.7), dpi=100)
    ax.imshow(z, origin='lower', extent=[0, W, 0, H], cmap='turbo', vmin=0, vmax=1.3,
              interpolation='bilinear')
    # border that matches the activation mask exactly (the ellipse used in bold_frame)
    ax.add_patch(Ellipse((cx * W, cy * H), 2 * rx * W, 2 * ry * H,
                         fill=False, edgecolor='#2a3346', lw=1.6))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    ax.text(0.02, 0.96, f't = {i}s', transform=ax.transAxes, fontsize=9, color='#2a3346',
            va='top')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, f'bold_{i:02d}.png'), transparent=True,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def render_movie(i):
    """A simple evolving scene: dusk sky shifting darker, a sun/moon descending,
    a car crossing left→right."""
    fig, ax = plt.subplots(figsize=(3.2, 1.8), dpi=100)
    tfrac = i / (N - 1)
    # sky gradient (day -> dusk): blue to indigo
    sky = np.linspace(0, 1, 100).reshape(-1, 1)
    top = np.array([0.20 + 0.1 * (1 - tfrac), 0.35 * (1 - tfrac) + 0.10, 0.55 - 0.2 * tfrac])
    bot = np.array([0.95 - 0.5 * tfrac, 0.55 - 0.3 * tfrac, 0.45 - 0.1 * tfrac])
    grad = (1 - sky) * top + sky * bot
    ax.imshow(grad.reshape(100, 1, 3), extent=[0, 10, 3, 10], aspect='auto')
    ax.add_patch(plt.Rectangle((0, 0), 10, 3.2, color=(0.12, 0.12, 0.16)))  # ground
    sun_x = 1 + 8 * tfrac
    sun_y = 8.5 - 4.5 * tfrac
    ax.scatter([sun_x], [sun_y], s=320, color=(1.0, 0.85 - 0.3 * tfrac, 0.5 - 0.3 * tfrac), zorder=3)
    car_x = -2 + 13 * tfrac
    ax.add_patch(plt.Rectangle((car_x, 1.0), 1.8, 0.8, color='#e0e0e6', zorder=4))
    ax.add_patch(plt.Rectangle((car_x + 0.3, 1.6), 1.1, 0.5, color='#cfd4e0', zorder=4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, f'movie_{i:02d}.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def render_waveform():
    """One static audio waveform strip for the whole clip (playhead drawn in JS)."""
    rng = np.random.RandomState(0)
    t = np.linspace(0, N, 1600)
    env = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * t / N))            # slow envelope
    wav = env * rng.normal(0, 1, t.size)
    fig, ax = plt.subplots(figsize=(8.0, 0.7), dpi=100)
    ax.fill_between(t, -np.abs(wav), np.abs(wav), color='#7c4dff', lw=0)
    ax.set_xlim(0, N); ax.set_ylim(-1.6, 1.6); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, 'waveform.png'), transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    outline, params = cortex_outline()
    for i in range(N):
        render_movie(i)
        render_bold(i, outline, params)
        print(f'  frame {i}: movie + BOLD', flush=True)
    render_waveform()
    print(f'wrote {N} movie+BOLD frames + waveform to {OUT}', flush=True)


if __name__ == '__main__':
    main()
