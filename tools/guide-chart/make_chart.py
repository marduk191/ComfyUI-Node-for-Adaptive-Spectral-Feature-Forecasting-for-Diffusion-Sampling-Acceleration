"""Generate the parameter reference chart (spectrum-patcher-guide.png).

    python tools/guide-chart/make_chart.py /tmp/guide.html
    node tools/guide-chart/render.js /tmp/guide.html spectrum-patcher-guide.png

render.js needs Playwright + Chromium and prints how far the content overlaps the footer
(must be <= 0). Fonts: Inter and JetBrains Mono.
"""
import html
import sys

OUT = sys.argv[1]

BLUE, GREEN, ORANGE, PURPLE, YELLOW = "#4f8cff", "#2fbf71", "#f08a24", "#a07cf0", "#e0b030"


def slider(name, default, rng, desc, ticks, left, right, hint, reverse=False):
    lo, hi = (ORANGE, GREEN) if reverse else (GREEN, ORANGE)
    tick_html = "".join(
        f'<div class="tick" style="left:{pos}%"><span style="color:{lo if pos < 20 else hi}">{html.escape(lbl)}</span></div>'
        for lbl, pos in ticks
    )
    grad = f"background:linear-gradient(90deg,{lo},#8fae3a 55%,{hi})"
    return f"""
<div class="card">
  <div class="pname">{name}</div>
  <div class="meta">default: {default} &nbsp; range: {rng}</div>
  <div class="desc">{desc}</div>
  <div class="track">
    <div class="row"><span class="end" style="color:{lo}">{left}</span><div class="bar" style="{grad}">{tick_html}</div><span class="end r" style="color:{hi}">{right}</span></div>
    <div class="hint">{hint}</div>
  </div>
</div>"""


def choice(name, default, rng, desc, boxes):
    box_html = "".join(
        f'<div class="box" style="border-color:{c};background:{c}22"><b style="color:{c}">{t}</b><small>{s}</small></div>'
        for t, s, c in boxes
    )
    return f"""
<div class="card">
  <div class="pname">{name}</div>
  <div class="meta">default: {default} &nbsp; {rng}</div>
  <div class="desc">{desc}</div>
  <div class="boxes">{box_html}</div>
</div>"""


cards = "".join([
    slider("window_size", "2", "1 – 32", "Real model pass every Nth step · forecast fills the skipped steps",
           [("1", 0), ("2", 28), ("4", 56)], "&lt; QUALITY", "FAST &gt;",
           "Lower = more real evaluations = higher fidelity"),
    slider("w", "0.50", "0.0 – 1.0", "Blend weight · 0 = last real output · 1 = pure forecast",
           [("0.3", 27), ("0.5", 48), ("0.8", 77)], "&lt; LAST OUTPUT", "FORECAST &gt;",
           "Start at 0.5; raise toward 0.8 if the output stays clean"),
    slider("m", "4", "1 – 16", "Chebyshev basis size · fit uses up to m+1 cached real steps",
           [("2", 7), ("4", 20), ("6", 33)], "&lt; SIMPLER", "FLEXIBLE &gt;",
           "Too high for the step count = overfitting · keep m &lt; steps / window_size"),
    slider("lam", "0.100", "0.0 – 100", "Ridge strength · pulls the forecast toward the last real output",
           [("0.01", 0), ("0.1", 9), ("1.0+", 38)], "&lt; TRUST FIT", "PLAIN CACHE &gt;",
           "Raise if forecasts drift · very high = plain reuse of the last output"),
    choice("model_type", "auto", "options: auto / eps / flow",
           "auto reads the model's own sampling type, the same check ComfyUI uses",
           [("auto", "all 104 ComfyUI models", BLUE),
            ("eps", "SD 1.5 · SDXL · SVD · PixArt", ORANGE),
            ("flow", "Flux · Wan · Qwen · Z-Image · SD3 · LTX", PURPLE)]),
    choice("real_final_step", "on", "optional",
           "Last step always runs the real model · its output becomes the final latent",
           [("on", "recommended · biggest quality win on 4–8 step flow runs", GREEN),
            ("off", "one more forecast step · lower quality", "#8a94a6")]),
    slider("warmup_steps", "0", "0 – 32 · optional", "First N steps always run the real model",
           [("0", 0), ("2", 25), ("5", 62)], "&lt; FAST", "QUALITY &gt;",
           "0 = minimal warm-up (2 real passes) · each extra step costs a model pass", reverse=True),
])

presets = [
    ("Max Quality", "1", "0.0", "4", "0.1", GREEN),
    ("Balanced", "2", "0.5", "4", "0.1", ORANGE),
    ("Fast", "4", "0.7", "3", "1.0", "#e05a2a"),
    ("Few-step flow<small>Z-Image Turbo · Flux schnell</small>", "2", "0.6", "4", "0.1", PURPLE),
    ("Long flow<small>Flux · Wan · Qwen · 20+ steps</small>", "3", "0.5", "4", "0.1", BLUE),
]
preset_rows = "".join(
    f'<tr><td class="sc" style="border-left-color:{c}">{n}</td><td>{a}</td><td>{b}</td><td>{d}</td><td>{e}</td></tr>'
    for n, a, b, d, e, c in presets
)

notes = [
    ("window_size=1", "no forecasting · identical to unpatched", BLUE),
    ("NaN / Inf", "auto-fallback to a real pass", GREEN),
    ("auto", "flow = CONST / COSMOS_RFLOW sampling", PURPLE),
    ("lam", "shrinks toward last output, not zero", ORANGE),
    ("m vs steps", "m &lt; steps / window_size", YELLOW),
]
note_rows = "".join(
    f'<div class="note" style="background:{c}1f"><b style="color:{c}">{k}</b><span>{v}</span></div>'
    for k, v, c in notes
)

guide = [("4–8 (Turbo)", "2", "3 – 4"), ("20", "2 – 3", "4 – 5"), ("50+", "4 – 6", "5 – 8")]
guide_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in guide)

page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ width: 1200px; height: 990px; background: #0d1117; color: #d7dde8; font-family: Inter, sans-serif; position: relative; overflow: hidden; }}
.mono, .pname, .meta, .tick span, td, .note b {{ font-family: 'JetBrains Mono', monospace; }}
header {{ height: 56px; border-bottom: 1px solid #222a36; padding: 10px 44px; display: flex; justify-content: space-between; align-items: center; }}
h1 {{ font-size: 25px; font-weight: 700; color: #f2f5fa; line-height: 1; }}
.sub {{ font-size: 12px; color: #7d8696; margin-top: 3px; }}
.pills {{ display: flex; gap: 16px; }}
.pill {{ font-size: 11px; font-weight: 700; padding: 6px 0; width: 84px; text-align: center; border: 1.5px solid; border-radius: 4px; }}
main {{ display: flex; gap: 34px; padding: 10px 36px 0 36px; }}
.left {{ width: 730px; display: flex; flex-direction: column; gap: 7px; }}
.card {{ background: #151b24; border-left: 3px solid {BLUE}; border-radius: 5px; padding: 8px 14px 9px 14px; }}
.pname {{ font-size: 17px; font-weight: 700; color: #f2f5fa; line-height: 1.1; }}
.meta {{ font-size: 10.5px; color: #6f7888; margin-top: 2px; }}
.desc {{ font-size: 11.5px; color: #aab3c2; margin-top: 2px; padding-bottom: 6px; border-bottom: 1px solid #222a36; }}
.track {{ position: relative; padding-top: 0; }}
.row {{ display: flex; align-items: center; gap: 10px; padding-top: 14px; }}
.end {{ font-family: 'JetBrains Mono', monospace; font-size: 8.5px; font-weight: 700; width: 78px; white-space: nowrap; }}
.end.r {{ text-align: right; }}
.bar {{ position: relative; flex: 1; height: 6px; border-radius: 3px; }}
.tick {{ position: absolute; top: -3px; width: 2px; height: 12px; background: #c9ced8aa; }}
.tick span {{ position: absolute; bottom: 13px; left: -12px; width: 26px; text-align: center; font-size: 10.5px; }}
.hint {{ text-align: right; font-size: 10.5px; color: #7d8696; margin-top: 4px; }}
.boxes {{ display: flex; gap: 10px; margin-top: 7px; }}
.box {{ flex: 1; border: 1.5px solid; border-radius: 5px; text-align: center; padding: 4px 6px 5px; }}
.box b {{ display: block; font-size: 15px; font-family: 'JetBrains Mono', monospace; }}
.box small {{ display: block; font-size: 10px; color: #aab3c2; margin-top: 1px; }}
.right {{ width: 342px; padding-top: 4px; }}
h2 {{ font-size: 11.5px; letter-spacing: 1.5px; color: #8a94a6; font-weight: 700; margin: 6px 0 9px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th {{ background: #1a212c; color: #aab3c2; font-size: 10.5px; font-weight: 600; padding: 9px 6px; text-align: center; font-family: Inter, sans-serif; }}
th:first-child {{ text-align: left; padding-left: 9px; }}
td {{ padding: 8px 6px; text-align: center; border-bottom: 1px solid #222a36; color: #e3e8f0; background: #131922; }}
td.sc {{ text-align: left; font-family: Inter, sans-serif; border-left: 3px solid; padding-left: 8px; font-size: 12px; }}
td.sc small {{ display: block; font-size: 9.5px; color: #7d8696; margin-top: 1px; }}
.grad {{ height: 4px; border-radius: 2px; background: linear-gradient(90deg, {GREEN}, {ORANGE}); margin: 22px 0 8px; }}
.gl {{ display: flex; justify-content: space-between; font-size: 9px; font-weight: 700; }}
.note {{ display: flex; gap: 8px; align-items: baseline; padding: 5px 8px; border-radius: 3px; margin-bottom: 4px; font-size: 10.5px; color: #aab3c2; }}
.note b {{ font-size: 10px; white-space: nowrap; }}
.guide {{ width: 300px; }}
footer {{ position: absolute; bottom: 0; left: 0; right: 0; height: 30px; border-top: 1px solid #222a36; padding: 8px 44px; display: flex; justify-content: space-between; font-size: 11px; color: #7d8696; }}
</style></head><body>
<header>
  <div><h1>Spectrum Patcher (Max)</h1><div class="sub">Parameter Reference · model/patches · ComfyUI</div></div>
  <div class="pills">
    <div class="pill" style="color:{GREEN};border-color:{GREEN};background:{GREEN}1a">QUALITY</div>
    <div class="pill" style="color:{ORANGE};border-color:{ORANGE};background:{ORANGE}1a">BALANCED</div>
    <div class="pill" style="color:#e05a2a;border-color:#e05a2a;background:#e05a2a1a">FAST</div>
  </div>
</header>
<main>
  <div class="left">{cards}</div>
  <div class="right">
    <h2>PRESETS</h2>
    <table><tr><th>Scenario</th><th>win</th><th>w</th><th>m</th><th>lam</th></tr>{preset_rows}</table>
    <div class="grad"></div>
    <div class="gl"><span style="color:{GREEN}">&lt; QUALITY</span><span style="color:{ORANGE}">SPEED &gt;</span></div>
    <div style="margin-top:8px">{note_rows}</div>
    <h2 style="margin-top:18px">STEP COUNT GUIDE</h2>
    <table class="guide"><tr><th>Steps</th><th>window_size</th><th>m (rec.)</th></tr>{guide_rows}</table>
    <div class="note" style="background:#2fbf711f;margin-top:10px"><b style="color:{GREEN}">all presets</b><span>model_type auto · real_final_step on · warmup 0</span></div>
  </div>
</main>
<footer><span>Falls back to a real model pass on NaN/Inf — safe to experiment with all values.</span><span>github.com/marduk191 · Spectrum Patcher (Max)</span></footer>
</body></html>"""

with open(OUT, "w") as f:
    f.write(page)
