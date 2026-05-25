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

HOW IT WORKS
-------------
All N oscillator slots are pre-baked into the Desmos state at startup as
tone(F[n], G[n]) expressions referencing two lists F and G (length = max_partials).
Each frame the player issues exactly ONE Calc.setExpressions() call that
updates both F and G as complete lists.  No expressions are ever created or
destroyed during playback — eliminating the per-expression parse/recompile
overhead that caused frame-rate degradation even at low partial counts.

PERFORMANCE
-----------
  Old approach: N setExpression calls/frame (one per changed slot) + Desmos
                re-evaluates the graph N times.
  New approach: 1 setExpressions call/frame with updated F+G lists. Desmos
                re-evaluates once. Scales to hundreds of partials with no
                frame-rate impact.

LIMITATIONS
-----------
  * Only channel 0 is exported (Desmos is mono).
  * Frequencies are clamped to 20-20000 Hz.
  * Amplitudes are globally normalised to 0-1.
  * tone() needs a user gesture — click the page before playback starts.
  * max_partials must stay ≤ ~500 to keep list size inside Desmos limits.
"""

import json
import argparse
from pathlib import Path

from numpy import add


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

def build_frames_js(data: dict, max_partials: int, channel: int = 0):
    """
    Build the JS frame data for single-call list playback.

    Each frame is stored as two flat arrays: F (frequencies) and G (gains),
    both of length max_partials.  Unused slots get freq=20, gain=0.

    The player issues exactly ONE Calc.setExpressions() per frame, updating
    the F and G list expressions.  No dynamic expression creation/removal.

    Output: RSC2_F and RSC2_G as JS arrays-of-arrays, indexed by frame.
    Only frames that differ from a silent frame are stored; others are null.
    """
    sr      = data["sample_rate"]
    hop     = data["hop_size"]
    ch_data = data["channels"][channel]
    frames  = ch_data["frames"]

    freq_min, freq_max = 20.0, 20_000.0
    hop_sec = hop / sr

    # Global peak for normalisation
    global_peak = max(
        (p["amp_norm"] * f["peak"] for f in frames for p in f["partials"]),
        default=1.0
    )
    if global_peak < 1e-12:
        global_peak = 1.0

    n_frames   = len(frames)
    all_F      = []   # list of F-arrays (or None for silent frames)
    all_G      = []   # list of G-arrays (or None for silent frames)

    for frame in frames:
        partials = [
            p for p in frame["partials"]
            if freq_min <= p["freq_hz"] <= freq_max
        ]
        partials.sort(key=lambda p: p["amp_norm"], reverse=True)
        partials = partials[:max_partials]

        if not partials:
            all_F.append(None)
            all_G.append(None)
            continue

        F = [20.0] * max_partials
        G = [0.0]  * max_partials
        for slot, p in enumerate(partials):
            F[slot] = round(max(freq_min, min(freq_max, p["freq_hz"])), 3)
            G[slot] = round(max(0.0, min(1.0, (p["amp_norm"] * frame["peak"]) / global_peak)), 6)

        all_F.append(F)
        all_G.append(G)

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
    }

    # Serialise as two sparse JS arrays (null for silent frames)
    js = (
        "const RSC2_F=" + json.dumps(all_F, separators=(",", ":")) + ";\n"
        "const RSC2_G=" + json.dumps(all_G, separators=(",", ":")) + ";\n"
    )

    active = sum(1 for f in all_F if f is not None)
    print(f"   Schedule: {n_frames} frames ({active} non-silent), "
          f"{max_partials} slots, 1 setExpressions call/frame")

    return js, meta


def build_desmos_state(meta: dict):
    """
    Pre-bake ALL oscillator expressions into the initial Desmos state.

    Structure:
      F = [20, 20, ...]   ← frequency list, length = max_partials
      G = [0,  0,  ...]   ← gain list,      length = max_partials
      tone(F[1], G[1])    ← one expression per slot (1-indexed in Desmos)
      tone(F[2], G[2])
      ...

    The player only ever calls setExpressions([{id:"F", latex:"F=[...]"},
    {id:"G", latex:"G=[...]"}]) — one call, constant cost regardless of
    how many partials are active.
    """
    freq_max     = meta["freq_max"]
    max_partials = meta["max_partials"]

    exprs     = []
    txt_lines = []

    def add(id_str, latex, folder=None, hidden=False, color="#00ff88"):
        e = make_expression(id_str, latex, hidden=hidden, color=color)
        if folder:
            e["folderId"] = folder
        exprs.append(e)
        txt_lines.append(f"  {id_str:12s}  {latex}")

    # ── Frequency and gain list variables ────────────────────────────────────
    zeros  = ",".join(["0"]  * max_partials)
    twenties = ",".join(["20"] * max_partials)
    exprs.append(make_folder("fold_data", "🔒 Audio Data", collapsed=True))
    add("F", f"F=[{twenties}]", color="#4488ff", folder="fold_data")
    add("G", f"G=[{zeros}]",   color="#ff8844", folder="fold_data")
    txt_lines.append("")

    # ── tone() expressions, one per slot ─────────────────────────────────────
    exprs.append(make_folder("fold_osc", f"🔊 Oscillators ({max_partials} slots)", collapsed=True))
    txt_lines.append(f"  fold_osc     🔊 Oscillators ({max_partials} slots)")

    add("audio", "\\operatorname{tone}\\left(F,G\\right)", color="#00ff88")

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
    state_json   = json.dumps(desmos_state)
    tick_ms      = meta["tick_ms"]
    n_frames     = meta["n_frames"]
    max_partials = meta["max_partials"]

    # Build the silent-frame F/G latex strings once (reused every silent frame)
    silent_F = "[" + ",".join(["20"] * max_partials) + "]"
    silent_G = "[" + ",".join(["0"]  * max_partials) + "]"

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
</div>
<div id="calculator"></div>
<script>
// RSC2 frame data — two arrays-of-arrays (null = silent frame)
{js_data}

// ── Desmos setup ──────────────────────────────────────────────────────────────
const Calc = Desmos.GraphingCalculator(
    document.getElementById('calculator'),
    {{ expressionList:true, settingsMenu:true, keypad:false, zoomButtons:true }}
);
Calc.setState({state_json});

// Pre-built silent latex strings (avoids rebuilding every silent frame)
const SILENT_F = "F={silent_F}";
const SILENT_G = "G={silent_G}";

// ── Frame application — exactly ONE setExpressions call per frame ─────────────
function applyFrame(fi) {{
    const F = RSC2_F[fi];
    const G = RSC2_G[fi];

    if (F !== null && G !== null) {{
        Calc.setExpressions([
            {{ id: "F", latex: "F=[" + F.join(",") + "]" }},
            {{ id: "G", latex: "G=[" + G.join(",") + "]" }},
        ]);
    }} else {{
        Calc.setExpressions([
            {{ id: "F", latex: SILENT_F }},
            {{ id: "G", latex: SILENT_G }},
        ]);
    }}

    document.getElementById('info').textContent         = `frame ${{fi + 1}} / {n_frames}`;
    document.getElementById('progress-bar').style.width = `${{(fi + 1) / {n_frames} * 100}}%`;
}}

// ── Playback control ──────────────────────────────────────────────────────────
let frameIdx = 0;
let timer    = null;
const TICK   = {tick_ms};
const N      = {n_frames};
let startTime = null;

function startPlayback() {{
    if (timer) return;
    document.getElementById('btn-play').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    frameIdx = 0;
    
    startTime = performance.now();

    function tick() {{
        const now = performance.now();
        const elapsed = now - startTime;
        
        // Calculate which frame should be playing RIGHT NOW
        targetFrame = Math.min(Math.floor(elapsed / TICK), N - 1);
        
        // Apply that frame (even if we skipped some)
        applyFrame(targetFrame);
        
        // Schedule next check for the upcoming frame boundary
        const nextFrameTime = startTime + (targetFrame + 1) * TICK;
        const delay = Math.max(0, nextFrameTime - now);
        
        if (targetFrame < N - 1) {{
            timer = setTimeout(tick, delay);
        }} else {{
            stopPlayback();
        }}
    }}
    tick();
}}

function stopPlayback() {{
    clearTimeout(timer);
    timer = null;
    Calc.setExpressions([
        {{ id: "F", latex: SILENT_F }},
        {{ id: "G", latex: SILENT_G }},
    ]);
    document.getElementById('btn-play').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}}

function seekTo(e) {{
    const t = (e.clientX - e.currentTarget.getBoundingClientRect().left) / e.currentTarget.offsetWidth;
    frameIdx = Math.max(0, Math.min(N - 1, Math.floor(t * N)));
    targetFrame = Math.max(0, Math.min(N - 1, Math.floor(t * N)));
    startTime = performance.now() - (targetFrame * TICK);
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
    ap.add_argument("input",          help="JSON output from rsc2_encode_for_desmos.py")
    ap.add_argument("--output",       help="Output base path (default: input stem)")
    ap.add_argument("--max-partials", type=int, default=192,
                    help="Max partials per frame (default 192)")
    ap.add_argument("--channel",      type=int, default=0,
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
