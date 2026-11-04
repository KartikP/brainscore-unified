"""Replay a saved clip beside its recorded temporal features, without inference."""
import base64
import io
import json

import numpy as np


def _presentation(assembly, stimulus_id):
    ids = np.asarray(assembly['stimulus_id'].values).astype(str)
    positions = np.flatnonzero(ids == stimulus_id)
    if len(positions) != 1:
        raise ValueError(f'Expected exactly one recording for {stimulus_id!r}; found {len(positions)}')
    return assembly.isel(presentation=int(positions[0]))


def _frame_bins(frame_count, fps, starts, ends):
    """Map frame centers to recorded intervals; preserve gaps instead of interpolating."""
    if frame_count < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('A clip needs frames and a finite positive frame rate')
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    if (starts.ndim != 1 or len(starts) == 0 or starts.shape != ends.shape
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.any(starts < 0) or np.any(ends <= starts)
            or np.any(starts[1:] < ends[:-1] - 1e-7)):
        raise ValueError('Recorded time bins must be finite, ordered and nonoverlapping')
    times = (np.arange(frame_count) + .5) * 1000 / fps
    return [int(found[0]) if len(found := np.flatnonzero((starts <= time) & (time < ends))) else -1
            for time in times]


def _payload(stimuli, neural, stimulus_id, behavior=None, display_units=64):
    from PIL import Image
    from brainscore.model_helpers.vlm_video_wrapper import read_clip
    from .experiment import parse_report

    if type(display_units) is not int or display_units < 1:
        raise ValueError('display_units must be a positive integer')
    rows = stimuli.loc[stimuli.stimulus_id.astype(str) == stimulus_id]
    if len(rows) != 1:
        raise ValueError('The selected stimulus must occur exactly once in the stimulus set')
    frames, fps = read_clip(rows.iloc[0].video_path)
    recording = _presentation(neural, stimulus_id)
    timing_error = ('Replay needs time-resolved activity with time_bin_start_ms and '
                    'time_bin_end_ms. It cannot invent timing for a static recording.')
    if set(recording.dims) != {'time_bin', 'neuroid'}:
        raise ValueError(timing_error)
    try:
        starts = np.asarray(recording['time_bin_start_ms'].values, dtype=float)
        ends = np.asarray(recording['time_bin_end_ms'].values, dtype=float)
    except KeyError:
        # BrainIO's MultiIndex level coordinates need not appear in list(coords).
        raise ValueError(timing_error) from None
    recording = recording.transpose('time_bin', 'neuroid')
    activity = np.asarray(recording.values, dtype=float)
    if activity.shape[1] == 0 or not np.isfinite(activity).all():
        raise ValueError('Recorded activity must contain finite unit values')
    mapping = _frame_bins(len(frames), fps, starts, ends)
    if not any(value >= 0 for value in mapping):
        raise ValueError('No recorded time bin overlaps the video frames')
    selected = np.linspace(0, activity.shape[1] - 1, min(display_units, activity.shape[1]), dtype=int)
    raw = activity[:, selected]
    std = raw.std(axis=0)
    normalized = np.divide(raw - raw.mean(axis=0), std, out=np.zeros_like(raw), where=std > 0)
    encoded = []
    for frame in frames:
        image = Image.fromarray(frame)
        image.thumbnail((360, 360))
        output = io.BytesIO()
        image.save(output, format='PNG')
        encoded.append('data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'))
    try:
        unit_ids = [str(value) for value in recording['neuroid_id'].values[selected]]
    except KeyError:
        unit_ids = [f'unit {index}' for index in selected]
    report, raw_answer, parse_error = None, '', None
    if behavior is not None:
        answer = _presentation(behavior, stimulus_id)
        try:
            raw_answer = str(answer['raw_answer'].item())
        except KeyError:
            raw_answer = str(answer.item())
        report, parse_error = parse_report(raw_answer)
    return {
        'stimulus_id': stimulus_id, 'fps': float(fps), 'frame_count': len(frames),
        'frames': encoded, 'frame_bins': mapping, 'start_ms': starts.tolist(), 'end_ms': ends.tolist(),
        'raw_activity': raw.tolist(), 'display_activity': normalized.tolist(),
        'unit_ids': unit_ids, 'total_units': activity.shape[1],
        'mean_absolute_activity': np.abs(activity).mean(axis=1).tolist(),
        'report': report, 'raw_answer': raw_answer, 'parse_error': parse_error,
    }


def replay_html(stimuli, neural, stimulus_id: str, *, behavior=None, display_units: int = 64) -> str:
    """Standalone replay HTML. Display normalization never alters the saved assembly.

    Use IPython.display.IFrame on an exported file, or an iframe with srcdoc.
    Time refers to the clip's recorded temporal patches, not computation latency
    or causal responses: a video model can use other frames to form each feature.
    """
    payload = _payload(stimuli, neural, stimulus_id, behavior, display_units)
    encoded = json.dumps(payload, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e')
    return _TEMPLATE.replace('__REPLAY_DATA__', encoded)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sun 2025 stimulus and activity replay</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:20px;font:14px/1.5 system-ui,sans-serif;color:#17243b;background:#f5f7fa}
main{max-width:1000px;margin:auto;background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px}
h1{font-size:19px;margin:0 0 5px}h2{font-size:14px;margin:0 0 8px}p{margin:6px 0}.muted{color:#526179;font-size:12px}
.grid{display:grid;grid-template-columns:minmax(220px,360px) minmax(280px,1fr);gap:22px;margin:20px 0}
canvas{max-width:100%;display:block}#clip{width:100%;border-radius:8px;background:#808080}
#heat{width:100%;height:300px;image-rendering:pixelated;cursor:crosshair}
.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:14px 0}
button,select{font:inherit;border:1px solid #bac8d8;background:white;padding:6px 12px;border-radius:6px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #5a96dd;outline-offset:2px}
button.primary{background:#164d85;color:white;border-color:#164d85}
input[type=range]{width:100%;accent-color:#164d85}.readout{font-variant-numeric:tabular-nums;min-height:22px}
#bin{font-weight:600}.legend{display:flex;align-items:center;gap:8px;margin:8px 0}
.gradient{width:110px;height:10px;background:linear-gradient(90deg,#3266af,#f7f7f7,#c74442)}
#trace{width:100%;height:90px}pre{white-space:pre-wrap;word-break:break-word;background:#f2f5f9;padding:10px;border-radius:6px}
@media(max-width:700px){.grid{grid-template-columns:1fr}body{padding:8px}main{padding:14px}}
</style></head><body><main>
<h1>Stimulus and recorded activity</h1><p id="identity" class="muted"></p>
<div class="grid">
<section><h2>Presented clip</h2><canvas id="clip" width="360" height="360" aria-label="Selected stimulus frame"></canvas>
<p id="clock" class="readout"></p><p class="muted">Video time, not model computation time.</p></section>
<section><h2 id="units"></h2><canvas id="heat" width="480" height="300" aria-label="Unit activity by recorded time bin"></canvas>
<div class="legend"><span class="muted">−2</span><span class="gradient"></span><span class="muted">+2 z</span></div>
<p class="muted">Rows: evenly sampled units. Columns: recorded bins. Colors are per-unit z-scores across this clip.</p>
<p id="bin" class="readout"></p><p id="hover" class="muted">Hover a cell to inspect its raw activation.</p></section>
</div>
<div class="controls"><button id="play" class="primary">Play</button><button id="restart">Restart</button>
<button id="previous" aria-label="Previous frame">Previous frame</button><button id="next" aria-label="Next frame">Next frame</button>
<label>Speed <select id="speed"><option value=".25">0.25×</option><option value=".5">0.5×</option>
<option value="1" selected>1×</option><option value="2">2×</option></select></label>
<label><input id="loop" type="checkbox"> Loop</label></div>
<label for="frame">Frame</label><input id="frame" type="range" min="0" step="1" value="0">
<h2>Mean absolute activity across all recorded units</h2><canvas id="trace" width="900" height="90"></canvas>
<p class="muted">Raw activation magnitude in model units; the vertical line follows video time.</p>
<details><summary>Model answer for this clip</summary><p id="answer"></p><pre id="raw"></pre></details>
<p class="muted">Replay reads saved measurements and runs no model inference. Values are held within each recorded bin;
missing bins remain missing. Temporal features may use information from the whole clip. They do not establish neural latency,
causal processing order, or biological brain activity. Display normalization leaves the recording unchanged.</p>
</main><script id="replay-data" type="application/json">__REPLAY_DATA__</script>
<script>
(() => {
const d=JSON.parse(document.getElementById('replay-data').textContent), el=id=>document.getElementById(id);
const clip=el('clip'), heat=el('heat'), trace=el('trace'), slider=el('frame');
const frames=d.frames.map(src=>{const im=new Image();im.src=src;return im;});
let frame=0,playing=false,request=null,base=0,started=0;
const duration=d.frame_count*1000/d.fps, count=d.start_ms.length, units=d.unit_ids.length;
slider.max=d.frame_count-1;
el('identity').textContent=d.stimulus_id+' · '+d.frame_count+' frames at '+d.fps+' fps';
el('units').textContent='Recorded activity: '+units+' of '+d.total_units+' units';
el('raw').textContent=d.raw_answer||'Behavior was not supplied.';
el('answer').textContent=d.report ? 'Direction: '+d.report.direction_deg+'°; speed: '+d.report.speed_px_per_frame+' pixels/frame'
 : (d.parse_error ? 'Invalid motion report: '+d.parse_error : 'No behavioral report.');
function color(z){
 const t=Math.min(Math.abs(z)/2,1),target=z<0?[50,102,175]:[199,68,66],zero=[247,247,247];
 return 'rgb('+zero.map((v,i)=>Math.round(v+(target[i]-v)*t)).join(',')+')';
}
function draw(){
 slider.value=frame;
 const bin=d.frame_bins[frame],time=frame*1000/d.fps,context=clip.getContext('2d');
 context.clearRect(0,0,clip.width,clip.height);
 if(frames[frame].complete && frames[frame].naturalWidth)context.drawImage(frames[frame],0,0,clip.width,clip.height);
 el('clock').textContent='Frame '+(frame+1)+' / '+d.frame_count+' · '+time.toFixed(1)+' ms';
 el('bin').textContent=bin<0?'No recorded activity for this frame':
  'Bin '+(bin+1)+' / '+count+' · '+d.start_ms[bin].toFixed(1)+'–'+d.end_ms[bin].toFixed(1)+' ms';
 const h=heat.getContext('2d'),cw=heat.width/count,ch=heat.height/units;
 h.clearRect(0,0,heat.width,heat.height);
 for(let c=0;c<count;c++)for(let r=0;r<units;r++){h.fillStyle=color(d.display_activity[c][r]);h.fillRect(c*cw,r*ch,Math.ceil(cw),Math.ceil(ch));}
 if(bin>=0){h.strokeStyle='#101f36';h.lineWidth=3;h.strokeRect(bin*cw+1.5,1.5,cw-3,heat.height-3);}
 const t=trace.getContext('2d'),values=d.mean_absolute_activity,max=Math.max(...values),min=Math.min(...values),span=max-min||1;
 t.clearRect(0,0,trace.width,trace.height);t.strokeStyle='#dce4ee';t.beginPath();t.moveTo(0,75);t.lineTo(trace.width,75);t.stroke();
 t.strokeStyle='#164d85';t.lineWidth=2;t.beginPath();
 values.forEach((value,i)=>{const y=70-(value-min)/span*55;
  t.moveTo(d.start_ms[i]/duration*trace.width,y);t.lineTo(d.end_ms[i]/duration*trace.width,y);});t.stroke();
 t.strokeStyle='#c74442';t.beginPath();t.moveTo(time/duration*trace.width,4);t.lineTo(time/duration*trace.width,78);t.stroke();
 t.fillStyle='#526179';t.font='11px system-ui';t.fillText(min.toPrecision(4)+'–'+max.toPrecision(4)+' model units',4,88);
}
function pause(){playing=false;if(request!==null)cancelAnimationFrame(request);request=null;el('play').textContent='Play';}
function tick(now){
 if(!playing)return;
 const position=base+(now-started)*d.fps/1000*Number(el('speed').value);
 if(position>=d.frame_count){
  if(el('loop').checked){base=0;started=now;frame=0;}
  else{frame=d.frame_count-1;draw();pause();return;}
 }else frame=Math.floor(position);
 draw();request=requestAnimationFrame(tick);
}
function play(){if(frame===d.frame_count-1)frame=0;playing=true;base=frame;started=performance.now();el('play').textContent='Pause';request=requestAnimationFrame(tick);}
el('play').onclick=()=>playing?pause():play();
el('restart').onclick=()=>{pause();frame=0;draw();};
el('previous').onclick=()=>{pause();frame=Math.max(0,frame-1);draw();};
el('next').onclick=()=>{pause();frame=Math.min(d.frame_count-1,frame+1);draw();};
slider.oninput=()=>{pause();frame=Number(slider.value);draw();};
el('speed').onchange=()=>{if(playing){pause();play();}};
heat.onmousemove=event=>{
 const rect=heat.getBoundingClientRect(),c=Math.min(count-1,Math.floor((event.clientX-rect.left)/rect.width*count)),
 r=Math.min(units-1,Math.floor((event.clientY-rect.top)/rect.height*units));
 el('hover').textContent=d.unit_ids[r]+' · bin '+(c+1)+' · raw '+d.raw_activity[c][r].toPrecision(6);
};
Promise.all(frames.map(im=>im.decode())).then(()=>{draw();document.body.dataset.ready='true';}).catch(error=>{
 el('clock').textContent='Could not decode clip frames: '+error.message;});
draw();
})();
</script></body></html>"""
