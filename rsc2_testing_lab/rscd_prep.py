#!/usr/bin/env python3
"""
RSC2 → Desmos Preprocessor  (streaming / unlimited edition)
=============================================================
Takes the JSON output of rsc2_encode_for_desmos.py and produces:

  1. player.html   — self-contained page that embeds the Desmos API and drives
                     playback by hot-swapping F_n / A_n each frame via
                     Calc.setExpression().  No list-size limit — all frame data
                     lives in JS, not inside Desmos.

  2. .desmos.json  — minimal Desmos state (just F_n=[], A_n=[], tone, viz).
                     Load via Calc.setState() in the console if you want to poke
                     at the expressions manually.

  3. .exprs.txt    — human-readable expression listing for debug.

HOW THE BYPASS WORKS
---------------------
Desmos has a ~10 000 element list limit *inside expressions*.  All frame data
lives in JS (no limit).  The Desmos JS API hot-swaps individual tone() gain
values each frame via setExpression().

GHOST OSCILLATOR PHASE TRICK
------------------------------
Desmos tone() always restarts its internal oscillator at phase 0 when the
expression changes frequency.  Normally this causes phase discontinuities
between frames — clicks and tonal smearing.

The fix: give every partial slot its OWN persistent tone() expression in
Desmos (one expression per oscillator slot, e.g. 192 expressions).  Each
expression's frequency never changes — it runs continuously.  We only ever
update the GAIN.

To get phase alignment, we pre-roll each oscillator silently (gain = GHOST)
N frames before it's needed, so that by the time the real frame arrives the
oscillator has naturally accumulated the exact phase the FFT measured:

    preroll_frames = round(target_phase / (2pi x freq x hop_sec))

    frame (real - preroll):   gain -> GHOST  (1e-6, inaudible, oscillator born)
    frame (real - preroll+1): gain -> GHOST  (still accumulating phase...)
    ...
    frame real:               gain -> amp    (SNAP -- phase-aligned!)
    frame real + duration:    gain -> GHOST  (back to ghost for next occurrence)

Since each oscillator slot has a fixed expression ID, Desmos never kills and
restarts it -- the Web Audio oscillator underneath runs continuously and
accumulates phase exactly as the FFT predicted.

LIMITATIONS
-----------
  * Only channel 0 is exported (Desmos is mono).
  * Frequencies are clamped to 20-20000 Hz.
  * Amplitudes are globally normalised to 0-1.
  * tone() needs a user gesture -- click the page before playback starts.
  * MAX_PREROLL caps pre-roll to avoid scheduling too far ahead.
"""

import json
import argparse
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_expression(id_str, latex, label=None, hidden=False, color="#00ff88"):
    expr = {
        "type":   "expression",
        "id":     id_str,
        "latex":  latex,
        "color":  color,
    }
    if label:
        expr["label"] = label
    if hidden:
        expr["hidden"] = True
    return expr


def make_folder(id_str, title, collapsed=True):
    return {
        "type":      "folder",
        "id":        id_str,
        "title":     title,
        "collapsed": collapsed,
    }


# ── Core builder ─────────────────────────────────────────────────────────────

GHOST_AMP  = 1e-6   # inaudible gain that keeps oscillator alive + accumulating phase
MAX_PREROLL = 32    # cap pre-roll so we don't schedule absurdly far ahead
TWO_PI     = 6.283185307179586


def build_frames_js(data: dict, max_partials: int, channel: int = 0):
    """
    Build the JS schedule for the ghost-oscillator phase trick.

    Each oscillator SLOT (0..max_partials-1) has a persistent tone() expression
    in Desmos.  Instead of changing frequency each frame (which resets phase),
    we keep frequency fixed and only update gain.

    For each partial in each frame we compute how many frames of silent pre-roll
    are needed so the oscillator's naturally-accumulated phase matches the FFT
    phase at the moment the real frame fires:

        phase_per_frame = 2pi * freq * hop_sec
        preroll = round(target_phase / phase_per_frame)  [clamped to MAX_PREROLL]

    Output: a JS object RSC2_SCHEDULE mapping frame_index -> list of
    {slot, freq, amp, preroll} events to apply at that frame.

    Also returns RSC2_FRAMES (raw data) for the spectrum visualiser.
    """
    import math

    sr       = data["sample_rate"]
    hop      = data["hop_size"]
    ch_data  = data["channels"][channel]
    frames   = ch_data["frames"]

    freq_min, freq_max = 20.0, 20_000.0
    hop_sec = hop / sr

    # Global peak for normalisation
    global_peak = max(
        (p["amp_norm"] * f["peak"] for f in frames for p in f["partials"]),
        default=1.0
    )
    if global_peak < 1e-12:
        global_peak = 1.0

    n_frames = len(frames)

    # schedule[frame_idx] = list of (slot, freq, amp) to SET at that frame
    # We build it as a dict of lists, then serialise
    schedule = {}   # frame_idx -> [(slot, freq, amp), ...]

    def sched(fi, slot, freq, amp):
        if fi < 0 or fi >= n_frames:
            return
        schedule.setdefault(fi, []).append((slot, freq, amp))

    # Also build raw frames for spectrum visualiser
    raw_frames = []

    for fi, frame in enumerate(frames):
        partials = [
            p for p in frame["partials"]
            if freq_min <= p["freq_hz"] <= freq_max
        ]
        partials.sort(key=lambda p: p["amp_norm"], reverse=True)
        partials = partials[:max_partials]

        raw_freqs = []
        raw_amps  = []

        for slot, p in enumerate(partials):
            freq  = round(max(freq_min, min(freq_max, p["freq_hz"])), 3)
            amp   = round(max(0.0, min(1.0, (p["amp_norm"] * frame["peak"]) / global_peak)), 6)
            phase = p["phase_rad"]   # -pi..pi from RSC2 encoder

            # How many frames of pre-roll to accumulate this phase?
            phase_per_frame = TWO_PI * freq * hop_sec
            if phase_per_frame > 1e-9:
                # Normalise phase to 0..2pi then find nearest integer frame count
                phase_norm  = (phase + TWO_PI) % TWO_PI
                preroll_raw = phase_norm / phase_per_frame
                preroll     = min(round(preroll_raw), MAX_PREROLL)
            else:
                preroll = 0

            # Schedule ghost start at (fi - preroll)
            ghost_start = fi - preroll
            if ghost_start >= 0:
                sched(ghost_start, slot, freq, GHOST_AMP)
            # If ghost_start < 0 we just start at fi with imperfect phase —
            # better than nothing for the first few frames

            # Schedule real amplitude snap at fi
            sched(fi, slot, freq, amp)

            # Schedule ghost-out one frame after (oscillator stays alive quietly)
            sched(fi + 1, slot, freq, GHOST_AMP)

            raw_freqs.append(freq)
            raw_amps.append(amp)

        raw_frames.append((raw_freqs, raw_amps))

    tick_ms  = round(hop_sec * 1000)
    tone_dur = round(hop_sec * 1.05, 6)

    meta = {
        "sample_rate":  sr,
        "hop_size":     hop,
        "hop_sec":      round(hop_sec, 6),
        "tone_dur":     tone_dur,
        "tick_ms":      tick_ms,
        "n_frames":     n_frames,
        "max_partials": max_partials,
        "freq_max":     freq_max,
        "ghost_amp":    GHOST_AMP,
    }

    # ── Serialise schedule as compact JS ─────────────────────────────────────
    # RSC2_SCHEDULE[fi] = [[slot,freq,amp], ...]  or undefined if nothing to do
    js_schedule = "const RSC2_SCHEDULE=" + json.dumps(
        {fi: [[s, f, a] for s, f, a in events]
         for fi, events in schedule.items()},
        separators=(",", ":")
    ) + ";\n"

    total_events = sum(len(v) for v in schedule.values())
    print(f"   Ghost-oscillator schedule: {n_frames} frames, "
          f"{total_events} gain events, {max_partials} oscillator slots")
    print(f"   Phase pre-roll: up to {MAX_PREROLL} frames lookahead 🎯")

    # We no longer emit raw frames for an in-Desmos visualiser — only the
    # per-frame schedule is needed for playback (reduces page size).
    return js_schedule, meta


def build_desmos_state(meta: dict):
    """
    Emit one persistent tone(freq_N, gain_N) expression per oscillator slot.
    JS updates only the gain variable each frame — frequency never changes so
    Desmos never resets the oscillator phase.
    """
    freq_max    = meta["freq_max"]
    max_partials = meta["max_partials"]
    ghost_amp   = meta["ghost_amp"]

    exprs     = []
    txt_lines = []

    def add(id_str, latex, label=None, hidden=False, color="#00ff88", folder=None):
        e = make_expression(id_str, latex, label=label, hidden=hidden, color=color)
        if folder:
            e["folderId"] = folder
        exprs.append(e)
        txt_lines.append(f"  {id_str:12s}  {latex}")

    # ── Ghost oscillator folder (start empty — we'll allocate dynamically)
    exprs.append(make_folder("fold_osc", f"👻 Ghost oscillators (dynamic)", collapsed=True))
    txt_lines.append("  Oscillators are created on-demand by the player JS and reclaimed when idle")

    graph = {
        "viewport":     {"xmin": 0, "xmax": freq_max, "ymin": -0.05, "ymax": 1.1},
        "xAxisLabel":   "Frequency (Hz)",
        "yAxisLabel":   "Amplitude",
        "showGrid":     True,
        "squareAxes":   False,
        "polarMode":    False,
        "xAxisNumbers": True,
        "yAxisNumbers": True,
    }
    state = {
        "version":     11,
        "graph":       graph,
        "expressions": {"list": exprs},
        "randomSeed":  "rsc2",
    }
    return state, "\n".join(txt_lines)


def build_player_html(js_data: str, desmos_state: dict, meta: dict) -> str:
        state_json = json.dumps(desmos_state)
        tick_ms = meta["tick_ms"]
        n_frames = meta["n_frames"]
        max_partials = meta["max_partials"]
        ghost_amp = meta["ghost_amp"]

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSC2 → Desmos Player</title>
<script src="https://www.desmos.com/api/v1.12/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
<style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0d0d1a;color:#e0e0ff;font-family:monospace;display:flex;flex-direction:column;height:100vh}}
    #header{{padding:10px 16px;background:#080810;border-bottom:1px solid #1a1a3a;display:flex;align-items:center;gap:14px}}
    h1{{font-size:13px;color:#00ff88;letter-spacing:3px;text-transform:uppercase;white-space:nowrap}}
    button{{background:#0d0d1a;border:1px solid #00ff88;color:#00ff88;padding:5px 12px;
                    font-family:monospace;font-size:12px;cursor:pointer;border-radius:2px;transition:all .1s}}
    button:hover{{background:#00ff88;color:#0d0d1a}}
    button:disabled{{opacity:.3;cursor:default;border-color:#333}}
    #progress{{flex:1;height:3px;background:#1a1a3a;border-radius:1px;cursor:pointer;position:relative}}
    #progress-bar{{height:100%;background:#00ff88;width:0%}}
    #info{{font-size:10px;color:#444;white-space:nowrap}}
    #phase-badge{{font-size:10px;color:#ff9900;border:1px solid #ff9900;padding:2px 6px;border-radius:2px;white-space:nowrap}}
    #calculator{{flex:1}}
</style>
</head>
<body>
<div id="header">
    <h1>⬡ RSC2/DESMOS</h1>
    <button id="btn-play" onclick="startPlayback()">▶ Play</button>
    <button id="btn-stop" onclick="stopPlayback()" disabled>■ Stop</button>
    <div id="progress" onclick="seekTo(event)"><div id="progress-bar"></div></div>
    <div id="info">frame 0 / {n_frames}</div>
    <div id="phase-badge">👻 PHASE GHOST</div>
</div>
<div id="calculator"></div>
<script>
// RSC2 data
{js_data}

// Desmos
const Calc = Desmos.GraphingCalculator(
    document.getElementById('calculator'),
    {{ expressionList:true, settingsMenu:true, keypad:false, zoomButtons:true }}
);
Calc.setState({state_json});

const N_SLOTS = {max_partials};
const GHOST = {ghost_amp};
const slotFreq = new Float64Array(N_SLOTS).fill(20);
const slotGain = new Float64Array(N_SLOTS).fill(GHOST);
// Track which logical slots have active expressions in Desmos
const slotCreated = new Uint8Array(N_SLOTS).fill(0);
const lastUsed = new Int32Array(N_SLOTS).fill(-1);
const MAX_IDLE_FRAMES = 64; // reclaim expressions after this many idle frames

// NOTE: per-frame schedule updates are batched below to reduce chattiness

// Playback
let frameIdx = 0;
let timer = null;
const TICK = {tick_ms};
const N = {n_frames};

function applyFrame(fi) {{
    const events = RSC2_SCHEDULE[fi];

    // Batch scheduled slot updates and send them in one call
    if (events) {{
        const batch = [];
        for (const ev of events) {{
            const [slot, freq, amp] = ev;

            // Create expressions on-demand if not present
            if (!slotCreated[slot]) {{
                slotCreated[slot] = 1;
                // Create f, g and tone() expression together
                batch.push({{ id: `osc_f${{slot}}`, latex: `f_{{${{slot}}}}=${{freq}}` }});
                batch.push({{ id: `osc_g${{slot}}`, latex: `g_{{${{slot}}}}=${{amp}}` }});
                // tone latex requires backslashes; build string via concatenation
                batch.push({{ id: `osc_t${{slot}}`, latex: "\\operatorname{{tone}}\\left(f_{{" + slot + "}},g_{{" + slot + "}}\\right)" }});
                slotFreq[slot] = freq;
                slotGain[slot] = amp;
            }} else {{
                if (slotFreq[slot] !== freq) {{
                    slotFreq[slot] = freq;
                    batch.push({{ id: `osc_f${{slot}}`, latex: `f_{{${{slot}}}}=${{freq}}` }});
                }}
                if (Math.abs(slotGain[slot] - amp) > 1e-9) {{
                    slotGain[slot] = amp;
                    batch.push({{ id: `osc_g${{slot}}`, latex: `g_{{${{slot}}}}=${{amp}}` }});
                }}
            }}

            lastUsed[slot] = fi;
        }}
        if (batch.length > 0) {{
            Calc.setExpressions(batch);
        }}
    }}

    // No visualiser: skip any F_n/A_n updates

    document.getElementById('info').textContent = `frame ${{fi+1}} / ${{N}}`;
    document.getElementById('progress-bar').style.width = `${{(fi+1)/N*100}}%`;

    // Reclaim unused slot expressions
    const toRemove = [];
    for (let i = 0; i < N_SLOTS; i++) {{
        if (slotCreated[i] && lastUsed[i] >= 0 && (fi - lastUsed[i]) > MAX_IDLE_FRAMES) {{
            slotCreated[i] = 0;
            toRemove.push(i);
        }}
    }}
    if (toRemove.length > 0) {{
        for (const i of toRemove) {{
            try {{
                Calc.removeExpression(`osc_f${{i}}`);
                Calc.removeExpression(`osc_g${{i}}`);
                Calc.removeExpression(`osc_t${{i}}`);
            }} catch (e) {{
                // ignore removal errors
            }}
        }}
    }}
}}

function startPlayback() {{
    if (timer) return;
    document.getElementById('btn-play').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    frameIdx = 0;
    function tick() {{
        if (frameIdx >= N) {{ stopPlayback(); return; }}
        applyFrame(frameIdx++);
        timer = setTimeout(tick, TICK);
    }}
    tick();
}}

function stopPlayback() {{
    clearTimeout(timer); timer = null;

    // Batch ghost all slots to avoid N separate Desmos calls
    const batch = [];
    for (let i = 0; i < N_SLOTS; i++) {{
        if (Math.abs(slotGain[i] - GHOST) > 1e-9) {{
            slotGain[i] = GHOST;
            batch.push({{ id: `osc_g${{i}}`, latex: `g_{{${{i}}}}=${{GHOST}}` }});
        }}
    }}
    if (batch.length > 0) Calc.setExpressions(batch);

    // Remove any allocated slot expressions to free Desmos expression count
    for (let i = 0; i < N_SLOTS; i++) {{
        if (slotCreated[i]) {{
            try {{
                Calc.removeExpression(`osc_f${{i}}`);
                Calc.removeExpression(`osc_g${{i}}`);
                Calc.removeExpression(`osc_t${{i}}`);
            }} catch (e) {{
                // ignore
            }}
            slotCreated[i] = 0;
            lastUsed[i] = -1;
            slotFreq[i] = 20;
            slotGain[i] = GHOST;
        }}
    }}

    document.getElementById('btn-play').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}}

function seekTo(e) {{
    const t = (e.clientX - e.currentTarget.getBoundingClientRect().left) / e.currentTarget.offsetWidth;
    frameIdx = Math.max(0, Math.min(N - 1, Math.floor(t * N)));
    applyFrame(frameIdx);
}}

applyFrame(0);
</script>
</body>
</html>
"""


# ── Clipboard / console helper ─────────────────────────────────────────────────

CONSOLE_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOW TO USE                                                                  ║
║                                                                              ║
║  OPTION A — player.html  (recommended, no list limit 🚀)                   ║
║    Open  {player_path}  in your browser.                                    ║
║    Click ▶ Play, then unmute Desmos (🔇 → 🔊 top-left of graph).           ║
║    Click the progress bar to seek.                                           ║
║                                                                              ║
║  OPTION B — raw Desmos state  (manual / debug)                              ║
║    1. Open  https://desmos.com/calculator                                   ║
║    2. F12 Console → run:                                                     ║
║         fetch('{json_path}')                                                ║
║           .then(r=>r.json()).then(s=>Calc.setState(s))                      ║
║    3. F_n and A_n start empty — you must drive them from JS yourself.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="RSC2 JSON → player.html + Desmos state (streaming, no list limit)")
    ap.add_argument("input",           help="JSON output from rsc2_encode_for_desmos.py")
    ap.add_argument("--output",        help="Output base path (default: input stem)")
    ap.add_argument("--max-partials",  type=int, default=192,
                    help="Max partials per frame (default 192, no Desmos list-size concern)")
    ap.add_argument("--channel",       type=int, default=0,
                    help="Which channel to export (default 0)")
    a = ap.parse_args()

    src  = Path(a.input)
    base = Path(a.output) if a.output else src.with_suffix("")

    print(f"📂 Loading {src} …")
    with open(src) as f:
        data = json.load(f)

    print(f"   {data['sample_rate']} Hz | {data['n_channels']} ch | "
          f"{data['n_frames']} frames | {data['duration_sec']:.2f}s")
    print(f"   Exporting ch {a.channel} | ALL {data['n_frames']} frames | "
          f"{a.max_partials} partials/frame")

    js_frames, meta = build_frames_js(data, max_partials=a.max_partials, channel=a.channel)
    state, txt      = build_desmos_state(meta)
    html            = build_player_html(js_frames, state, meta)

    player_path = base.with_suffix(".player.html")
    desmos_path = base.with_suffix(".desmos.json")
    exprs_path  = base.with_suffix(".exprs.txt")

    player_path.write_text(html)
    with open(desmos_path, "w") as f:
        json.dump(state, f, indent=2)
    with open(exprs_path, "w") as f:
        f.write("RSC2 → Desmos  expression listing\n")
        f.write("=" * 60 + "\n\n")
        f.write(txt + "\n")

    print(f"\n✅ Output files:")
    print(f"   🌐 {player_path}   ({player_path.stat().st_size/1024:.0f} KB)  ← open this!")
    print(f"   📊 {desmos_path}   ({desmos_path.stat().st_size/1024:.1f} KB)")
    print(f"   📝 {exprs_path}")
    print(CONSOLE_INSTRUCTIONS.format(
        player_path=player_path.resolve(),
        json_path=desmos_path.resolve()))


if __name__ == "__main__":
    main()