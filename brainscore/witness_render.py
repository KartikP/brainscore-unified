"""Render a WitnessEvent into a single "what the model saw / did" panel.

Modality-aware: vision/video/audio stimuli become image thumbnails, text becomes
a text card, an embodied step shows the rendered frame, a state-change shows the
lesion target. The caption always reports the mode, the instruction (if any), the
recording target (if neural), and a summary of what the model produced.

Kept separate from ``witness.py`` so the recorder stays dependency-light; this
module imports matplotlib/PIL only when a panel is actually rendered.
"""

_INK = '#1b2333'
_MUTED = '#5b6678'
_MODE_COLOR = {'neural': '#2f6bff', 'readout': '#1f9d57', 'generation': '#7c4dff',
               'action': '#c6810f', 'state_change': '#d8483b'}


def _did_caption(did):
    if not did:
        return ''
    kind = did.get('kind', '')
    if kind == 'action':
        return f"action → {did.get('action')}"
    if kind == 'neural':
        bits = []
        if 'region' in did:
            bits.append("region " + ", ".join(did['region']))
        if 'layer' in did:
            bits.append("layer " + ", ".join(did['layer']))
        bits.append(f"assembly {did.get('shape')}")
        if 'value_sample' in did:
            bits.append("e.g. " + ", ".join(str(v) for v in did['value_sample'][:3]))
        return " · ".join(bits)
    if kind == 'behavior':
        if 'prediction_sample' in did:
            return "predicted → " + ", ".join(did['prediction_sample'][:5])
        if 'labels' in did and 'value_sample' in did:
            pairs = list(zip(did['labels'], did['value_sample']))
            pairs = ", ".join(f"{l}={v}" for l, v in pairs[:5])
            return "P(label): " + pairs
        return f"behavior {did.get('shape')}"
    return f"{kind} {did.get('shape', '')}".strip()


def render_event(ev, out_path, label=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    saw = ev.saw or []
    # collect renderable image paths / frames
    images = []   # list of (kind, payload)
    for s in saw:
        if '_frame' in s:
            images.append(('frame', s['_frame']))
        for mod in ('vision', 'video', 'audio'):
            if mod in s and isinstance(s[mod], dict) and 'path' in s[mod]:
                images.append((mod, s[mod]['path']))
    texts = [s['text']['text'] for s in saw if 'text' in s and isinstance(s['text'], dict)]

    n_imgs = min(len(images), 4)
    has_text = bool(texts) and n_imgs == 0
    is_sc = ev.event_type == 'state_change'

    ncols = max(1, n_imgs)
    fig_w = max(5.0, 2.6 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2))
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    mode_c = _MODE_COLOR.get(ev.mode, _INK)

    if n_imgs > 0:
        for ax, (kind, payload) in zip(axes, images[:n_imgs]):
            try:
                if kind == 'frame':
                    ax.imshow(np.asarray(payload))
                else:
                    from PIL import Image
                    ax.imshow(Image.open(payload).convert('RGB'))
            except Exception as e:
                ax.text(0.5, 0.5, f'<{kind}>\n{e}', ha='center', va='center', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#dde3ee')
    else:
        ax = axes[0]
        ax.axis('off')
        if is_sc:
            s = saw[0] if saw else {}
            body = (f"STATE CHANGE\n\nkind: {s.get('kind')}\n"
                    f"target: {s.get('target')}\nperturbation: {s.get('perturbation')}")
        elif has_text:
            body = "TEXT INPUT\n\n" + "\n\n".join('“' + t[:240] + ('…' if len(t) > 240 else '') + '”'
                                                  for t in texts[:3])
        else:
            body = "(no renderable input)"
        ax.text(0.02, 0.98, body, ha='left', va='top', fontsize=11, color=_INK, wrap=True,
                family='sans-serif')

    # title + caption
    title = f"{label}   ·   call #{ev.index}   ·   {ev.mode.upper()}   ·   {ev.event_type}"
    fig.suptitle(title, fontsize=12, color=mode_c, y=0.99)

    cap_lines = []
    if ev.instruction:
        cap_lines.append(f'instruction: “{ev.instruction[:120]}”')
    if ev.recording_target:
        cap_lines.append('recording: ' + ", ".join(ev.recording_target))
    if ev.n_inputs:
        cap_lines.append(f'{ev.n_inputs} input(s); modalities: {", ".join(ev.modalities) or "—"}')
    did = _did_caption(ev.did)
    if did:
        cap_lines.append('DID: ' + did)
    if ev.error:
        cap_lines.append('ERROR: ' + ev.error)
    caption = '\n'.join(cap_lines)
    fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9, color=_MUTED)

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(out_path, dpi=130, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out_path
