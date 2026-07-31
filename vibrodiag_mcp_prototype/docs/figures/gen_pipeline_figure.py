#!/usr/bin/env python3
"""Generate the native-vibration pipeline block-diagram figure (SVG)."""
from __future__ import annotations

# ---------------------------------------------------------------- palette
INK = "#1A2422"
MUTED = "#5A6965"
LINE = "#C7D2CE"
CARD = "#FFFFFF"
SUBCARD = "#F7FAF9"
BG = "#FFFFFF"
ACCENT = "#0E7C5B"
ACCENT_DARK = "#0A5C44"
ACCENT_SOFT = "#E3F2EC"
AMBER = "#96660F"
AMBER_SOFT = "#F6EAD2"
RAIL = "#7E968F"
CODEBG = "#F0F4F2"

SANS = "DejaVu Sans, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"

W = 1240
X0, X1 = 60, 1080          # stage card span
CX = (X0 + X1) // 2        # connector center
PAD = 20                   # card inner padding
NUMW = 34                  # stage-number square
CONTX = X0 + PAD + NUMW + 16  # content left edge inside a card
CONTW = X1 - PAD - CONTX      # content width
RAILX = 1148               # feedback rail x

out: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s: str, size: float, mono: bool = False, bold: bool = False) -> float:
    """approx text width"""
    k = 0.602 if mono else (0.72 if bold else 0.52)
    return len(s) * size * k


def text(x, y, s, size=11.5, fill=INK, mono=False, bold=False, anchor="start", spacing=None):
    fam = MONO if mono else SANS
    w = "600" if bold else "400"
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    sp = ' xml:space="preserve"' if s.startswith(" ") or "  " in s else ""
    out.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{ls}{sp}>{esc(s)}</text>'
    )


def rrect(x, y, w, h, r=6, fill=CARD, stroke=LINE, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def hline(x0, y, x1, stroke=LINE, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def badge(x, y, label, style):
    """style: new | same | meas | judg ; returns width"""
    size = 9.0
    w = tw(label, size, mono=True) + 14
    h = 15
    if style == "new":
        rrect(x, y - h + 3, w, h, r=3, fill=ACCENT, stroke=ACCENT)
        text(x + 7, y - 1, label, size=size, fill="#FFFFFF", mono=True, spacing="0.08em")
    elif style == "same":
        rrect(x, y - h + 3, w, h, r=3, fill="none", stroke=LINE)
        text(x + 7, y - 1, label, size=size, fill=MUTED, mono=True, spacing="0.08em")
    elif style == "meas":
        rrect(x, y - h + 3, w, h, r=3, fill=ACCENT_SOFT, stroke=ACCENT_SOFT)
        text(x + 7, y - 1, label, size=size, fill=ACCENT_DARK, mono=True, spacing="0.08em")
    else:
        rrect(x, y - h + 3, w, h, r=3, fill=AMBER_SOFT, stroke=AMBER_SOFT)
        text(x + 7, y - 1, label, size=size, fill=AMBER, mono=True, spacing="0.08em")
    return w


def chip(x, y, label, ref=False, mono=True, size=10.5):
    w = tw(label, size, mono=mono) + 16
    h = 21
    rrect(x, y, w, h, r=4, fill=SUBCARD, stroke=(ACCENT if ref else LINE), sw=1.2 if ref else 1)
    text(x + 8, y + 14.5, label, size=size, fill=(ACCENT_DARK if ref else INK), mono=mono)
    return w


def stage_header(x, y, num, title, badges):
    # number square
    rrect(x + PAD, y + PAD - 4, NUMW, NUMW, r=6, fill=SUBCARD, stroke=LINE)
    text(x + PAD + NUMW / 2, y + PAD + 18.5, str(num), size=14, fill=ACCENT_DARK, mono=True, bold=True, anchor="middle")
    tx = CONTX
    text(tx, y + PAD + 12, title, size=14.5, bold=True)
    bx = tx + tw(title, 14.5, bold=True) + 12
    for label, style in badges:
        bx += badge(bx, y + PAD + 12, label, style) + 6


def connector(y0, y1, label=None, label2=None):
    out.append(f'<line x1="{CX}" y1="{y0:.1f}" x2="{CX}" y2="{y1 - 7:.1f}" stroke="{RAIL}" stroke-width="1.6"/>')
    out.append(f'<path d="M {CX - 5} {y1 - 8:.1f} L {CX + 5} {y1 - 8:.1f} L {CX} {y1:.1f} Z" fill="{RAIL}"/>')
    if label:
        text(CX + 14, (y0 + y1) / 2 - (4 if label2 else -3), label, size=10.5, fill=MUTED, mono=True)
    if label2:
        text(CX + 14, (y0 + y1) / 2 + 9, label2, size=10.5, fill=MUTED, mono=True)


def mini_arrow(x0, y, x1):
    out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1 - 6:.1f}" y2="{y:.1f}" stroke="{RAIL}" stroke-width="1.4"/>')
    out.append(f'<path d="M {x1 - 7:.1f} {y - 4:.1f} L {x1 - 7:.1f} {y + 4:.1f} L {x1:.1f} {y:.1f} Z" fill="{RAIL}"/>')


def sub(x, y, w, h, title, lines, mono_lines=(), title_fill=INK):
    rrect(x, y, w, h, r=5, fill=SUBCARD, stroke=LINE)
    text(x + 12, y + 18, title, size=11.5, bold=True, fill=title_fill)
    yy = y + 34
    for ln in lines:
        text(x + 12, yy, ln, size=10.5, fill=MUTED)
        yy += 14.5
    for ln in mono_lines:
        text(x + 12, yy, ln, size=10, fill=INK, mono=True)
        yy += 14


# ================================================================ header
H = 2312
out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="{SANS}">'
)
out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

text(X0, 64, "VIBROAGENT · TARGET ARCHITECTURE · NATIVE_VIBRATION_LLM_PLAN.MD §2–§2.1", size=10, fill=ACCENT_DARK, mono=True, spacing="0.12em")
text(X0, 94, "Native vibration pipeline — runtime, one 10 s check window", size=23, bold=True)
text(X0, 118, "After Phases 2–6: the on-device LLM reads waveforms as learned discrete tokens through the ordinary text path.", size=12, fill=MUTED)
text(X0, 135, "Shim, grammar enforcement, registry plumbing and the LLM-only policy are unchanged.", size=12, fill=MUTED)

# constraint chips
cx = X0
for label in ["HTP0 n_ctx 6144", "closed C plugin → text tokens only", "no hosted-model distillation", "LLM-only decisions"]:
    cx += chip(cx, 152, label) + 8

# legend
lx = X0
ly = 196
lx += badge(lx, ly, "NEW", "new") + 4
text(lx, ly - 1, "built in Phases 2–6", size=10, fill=MUTED); lx += tw("built in Phases 2–6", 10) + 18
lx += badge(lx, ly, "UNCHANGED", "same") + 4
text(lx, ly - 1, "existing plumbing", size=10, fill=MUTED); lx += tw("existing plumbing", 10) + 18
lx += badge(lx, ly, "MEASUREMENT", "meas") + 4
text(lx, ly - 1, "deterministic, allowed", size=10, fill=MUTED); lx += tw("deterministic, allowed", 10) + 18
lx += badge(lx, ly, "JUDGMENT", "judg") + 4
text(lx, ly - 1, "the model's alone", size=10, fill=MUTED)

# ================================================================ stage 1
y = 216
h1 = 130
rrect(X0, y, X1 - X0, h1, r=8)
stage_header(X0, y, 1, "SENSE — synchronized read", [("UNCHANGED", "same")])
sx = CONTX
sx += chip(sx, y + 44, "baseline (reference)", ref=True) + 8
for i in range(1, 6):
    sx += chip(sx, y + 44, f"target_{i}") + 8
text(CONTX, y + 88, "6× STWIN.box · IIS3DWB MEMS accelerometer · 26.7 kHz ODR · axes x/y/z · wall-aligned cross-sensor windows (±0.25 s)", size=10.5, fill=MUTED)
text(CONTX, y + 105, "via SDK vibrometer read — raw sample arrays never leave the process and never enter a prompt", size=10.5, fill=MUTED)

yc = y + h1
connector(yc, yc + 40, "6 sensors × 3 axes × 267,000 samples")

# ================================================================ stage 2
y = yc + 40
h2 = 170
rrect(X0, y, X1 - X0, h2, r=8)
stage_header(X0, y, 2, "DECIMATE", [("NEW", "new"), ("MEASUREMENT", "meas")])
bw = (CONTW - 12) / 2
sub(CONTX, y + 40, bw, 108, "Anti-alias + polyphase resample",
    ["26.7 kHz → 400 Hz (ratio 1/66.75), per axis",
     "keeps ≤ 200 Hz; building band of interest ≤ ~100 Hz",
     "high-pass removes DC / gravity component first"])
sub(CONTX + bw + 12, y + 40, bw, 108, "Amplitude normalize (per sensor-window)",
    ["codec sees waveform shape only — scale-invariant (§9)",
     "absolute level exits here as one text scalar per sensor",
     "(rms in g, joins the prompt context in stage 4)"])

yc = y + h2
connector(yc, yc + 40, "6 × 3 × 4,000 samples @ 400 Hz", "+ 6 level scalars")

# ================================================================ stage 3
y = yc + 40
h3 = 226
rrect(X0, y, X1 - X0, h3, r=8)
stage_header(X0, y, 3, "ENCODE — VQ codec (“the eye”)", [("NEW", "new"), ("MEASUREMENT", "meas")])
bw3 = (CONTW - 2 * 46) / 3
sub(CONTX, y + 40, bw3, 92, "1-D conv encoder",
    ["tri-axis in channels", "stride ↓32 → 12.5 frames/s", "~1–4 M params total"])
mini_arrow(CONTX + bw3 + 6, y + 86, CONTX + bw3 + 40)
sub(CONTX + bw3 + 46, y + 40, bw3, 92, "Residual VQ",
    ["2 codebooks × 1024 entries", "code dim ~64", "usage-entropy health checks"])
mini_arrow(CONTX + 2 * bw3 + 52, y + 86, CONTX + 2 * bw3 + 86)
sub(CONTX + 2 * (bw3 + 46), y + 40, bw3, 92, "Codes out",
    ["125 frames × 2 codebooks", "= 250 codes / sensor-window", "≤ 25 codes per signal-second"])
text(CONTX, y + 152, "codec_fit — reconstruction error per window, quantized {good | fair | poor}: a deterministic novelty measurement fed to the model as input", size=10.5, fill=MUTED)
text(CONTX, y + 169, "trained self-supervised on the .dat archive + public corpora + synthetically perturbed windows — then FROZEN + versioned (it is the token dictionary)", size=10.5, fill=MUTED)
text(CONTX, y + 186, "hard gate (R2): injected staged-event signatures — tones, impulse ring-downs — must visibly survive the encode→decode round-trip", size=10.5, fill=MUTED)
text(CONTX, y + 205, "decoder used offline only (training + codec_fit); runtime path is encoder → codes", size=10, fill=INK, mono=True)

yc = y + h3
connector(yc, yc + 40, "6 × 250 discrete codes + codec_fit")

# ================================================================ stage 4
y = yc + 40
h4 = 322
S4_MID = y + 92  # aim the registry-feedback arrow at the few-shot retrieval block
rrect(X0, y, X1 - X0, h4, r=8)
stage_header(X0, y, 4, "TOKENIZE + ASSEMBLE PROMPT", [("NEW", "new")])
bw4 = (CONTW - 12) / 2
sub(CONTX, y + 40, bw4, 104, "Code → token mapping (rung B)",
    ["(codebook, index) → 2,048 reserved Qwen3 vocab ids",
     "1 token per code · zero structural change to the GGUF",
     "fallback rung C: plain-text codes “v017 w3a2 …” (~2–3 tok/code)"])
sub(CONTX + bw4 + 12, y + 40, bw4, 104, "Few-shot retrieval from registry",
    ["nearest confirmed past event by code-space similarity",
     "its actual codes + confirmed label ≈ 300 tok",
     "in-context learning on real signals (one encounter per fleet)"])
# budget bar
by = y + 172
text(CONTX, by - 10, "context budget (n_ctx 6144):", size=10.5, fill=MUTED, mono=True)
segs = [
    (450, ACCENT, 0.55, "instructions+schema ≈450"),
    (260, ACCENT, 1.0, "ref codes ≈260"),
    (1290, ACCENT, 0.78, "5× target codes ≈1290"),
    (200, ACCENT, 0.35, "context ≈200"),
    (300, AMBER, 0.8, "exemplar ≈300"),
    (250, AMBER, 0.45, "output ≤250"),
    (3394, "#EDF2F0", 1.0, "headroom 3,394"),
]
total = 6144
bx = CONTX
bw_all = CONTW
for tok, color, op, _ in segs:
    wseg = bw_all * tok / total
    out.append(f'<rect x="{bx:.1f}" y="{by}" width="{wseg:.1f}" height="22" fill="{color}" fill-opacity="{op}"/>')
    bx += wseg
rrect(CONTX, by, bw_all, 22, r=3, fill="none", stroke=LINE)
# key
kx, ky = CONTX, by + 40
for tok, color, op, label in segs:
    out.append(f'<rect x="{kx:.1f}" y="{ky - 9}" width="9" height="9" rx="2" fill="{color}" fill-opacity="{op}" stroke="{LINE}" stroke-width="0.5"/>')
    text(kx + 13, ky, label, size=9.5, fill=MUTED, mono=True)
    kx += 13 + tw(label, 9.5, mono=True) + 16
    if kx > X1 - 220:
        kx = CONTX; ky += 17
text(CONTX, ky + 24, "compact text context = sensor ids, locations, level scalars (rms g), codec_fit, duration check — worst case ≈ 2.7–3.8 k of 6,144 tokens", size=10.5, fill=MUTED)
text(CONTX, ky + 41, "headroom reserved for a second time window (trend) or a second exemplar", size=10.5, fill=MUTED)

yc = y + h4
connector(yc, yc + 40, "one chat request ≈ 2.7–3.8 k tokens")

# ================================================================ stage 5
y = yc + 40
h5 = 196
S5_MID = y + h5 / 2
rrect(X0, y, X1 - X0, h5, r=8)
stage_header(X0, y, 5, "INFER ON THE NPU", [("UNCHANGED", "same"), ("NEW WEIGHTS", "new")])
bw5 = (CONTW - 46) / 2
sub(CONTX, y + 40, bw5, 112, "GenieX shim  :18181",
    ["token-count guard — over-budget prompts rejected loudly",
     "json_schema compiled → GBNF grammar (enforced decode)",
     "single-flight lock · 180 s generation watchdog → re-exec",
     "OpenAI-compatible surface, unchanged for all callers"])
mini_arrow(CONTX + bw5 + 6, y + 96, CONTX + bw5 + 40)
sub(CONTX + bw5 + 46, y + 40, bw5, 112, "Qwen3-4B-Instruct + LoRA",
    ["2,048 new input embedding rows trained (input-only;",
     "lm_head rows for vibration tokens stay masked)",
     "merge → GGUF → Q4_0 · llama.cpp HTP0 · n_ctx 6144",
     "~11 tok/s decode · 4B is the measured HTP0 ceiling (§10)"])
text(CONTX, y + 172, "gate first (R1): the closed HTP plugin must load the modified GGUF — rung B keeps the file structurally identical; rung C needs no tokenizer change at all", size=10.5, fill=MUTED)

yc = y + h5
connector(yc, yc + 40, "grammar-constrained JSON ≤ 250 tok")

# ================================================================ stage 6
y = yc + 40
h6 = 400
rrect(X0, y, X1 - X0, h6, r=8)
stage_header(X0, y, 6, "VERDICT — describe → decide → explain", [("JUDGMENT", "judg")])
# JSON block
jx, jy, jw, jh = CONTX, y + 40, CONTW, 196
rrect(jx, jy, jw, jh, r=5, fill=CODEBG, stroke=LINE)
mono_rows = [
    ('{ "sensor_reports": [ {', INK, None),
    ('    "sensor_id": "target_3",', INK, None),
    ('    "amplitude_change":  "large",', ACCENT_DARK, "describe · enum: none | mild | large"),
    ('    "spectral_character":"new_tone",', ACCENT_DARK, "enum: unchanged | new_tone | broadband_rise | shifted"),
    ('    "impulsiveness":     "none",', ACCENT_DARK, "enum: none | single | repeated"),
    ('    "persistence":       "sustained",', ACCENT_DARK, "enum: transient | sustained | recurring"),
    ('    "severity":          "significant" } ],', AMBER, "decide · enum: none | mild | significant — pipeline branches on this"),
    ('  "spatial_extent": "local",', AMBER, "enum: none | local | multi | building_wide"),
    ('  "affected_sensor_ids": ["target_3"],', AMBER, "filtered to real ids — hallucinated ids cannot enter"),
    ('  "confidence": 0.62,', INK, "model's own number — no numeric anchor in prompt"),
    ('  "explanation": "…" }', MUTED, "call 2 · hedged prose · never parsed for actions"),
]
ry = jy + 22
for code, color, comment in mono_rows:
    text(jx + 14, ry, code, size=10.5, fill=color, mono=True)
    if comment:
        text(jx + 375, ry, "// " + comment, size=9.5, fill=MUTED, mono=True)
    ry += 15.5
text(jx, jy + jh + 18, "grammar field order forces the axes before severity — perception before verdict; the constrained JSON doubles as chain-of-thought (§2.1)", size=10.5, fill=MUTED)
# call blocks
cby = jy + jh + 32
bw6 = (CONTW - 24) / 3
sub(CONTX, cby, bw6, 96, "Call 1 — classify (fast path)",
    ["k = 1, greedy decode", "labels-only JSON ≤ 250 tok", "every scheduled check"])
sub(CONTX + bw6 + 12, cby, bw6, 96, "Escalate on any deviation",
    ["vote_k = 5 sampled re-runs", "(top-k/top-p opened, seeded)", "→ vote-share confidence", "+ adversarial self-check call"])
sub(CONTX + 2 * (bw6 + 12), cby, bw6, 96, "Call 2 — explain (on demand)",
    ["verbalizes from labels + metrics", "cause speculation stays hedged", "operator prose, never parsed"])

yc = y + h6
connector(yc, yc + 40, "severity + affected ids drive actions;", "axes + prose accompany them")

# ================================================================ stage 7
y = yc + 40
h7 = 216
rrect(X0, y, X1 - X0, h7, r=8, dash=None)
stage_header(X0, y, 7, "ACT + LEARN", [("UNCHANGED", "same")])
sx = CONTX
for label in ["monitor loop", "anomaly registry (jsonl)", "graph-page anomalies panel", "webchat"]:
    sx += chip(sx, y + 42, label) + 8
ret_y = y + 82
bw7 = (CONTW - 12) / 2
rrect(CONTX, ret_y, bw7, 96, r=5, fill=SUBCARD, stroke=RAIL, dash="4 3")
text(CONTX + 12, ret_y + 19, "Exemplar loop — immediate", size=11.5, bold=True)
text(CONTX + 12, ret_y + 37, "human confirms/rejects in registry → confirmed event's", size=10.5, fill=MUTED)
text(CONTX + 12, ret_y + 51.5, "actual codes become retrievable few-shot material at once;", size=10.5, fill=MUTED)
text(CONTX + 12, ret_y + 66, "exemplars shareable fleet-wide", size=10.5, fill=MUTED)
text(CONTX + 12, ret_y + 84, "↩ feeds stage 4 retrieval", size=10, fill=ACCENT_DARK, mono=True)
rrect(CONTX + bw7 + 12, ret_y, bw7, 96, r=5, fill=SUBCARD, stroke=RAIL, dash="4 3")
text(CONTX + bw7 + 24, ret_y + 19, "Weight loop — periodic", size=11.5, bold=True)
text(CONTX + bw7 + 24, ret_y + 37, "confirmations append to the SFT set → LoRA refresh every", size=10.5, fill=MUTED)
text(CONTX + bw7 + 24, ret_y + 51.5, "few hundred confirmations (offline GPU) → re-quantize Q4_0", size=10.5, fill=MUTED)
text(CONTX + bw7 + 24, ret_y + 66, "→ redeploy; legacy flat labels stay as a derived A/B view", size=10.5, fill=MUTED)
text(CONTX + bw7 + 24, ret_y + 84, "↩ updates stage 5 weights", size=10, fill=ACCENT_DARK, mono=True)

# ---------------- feedback rail (dashed, right side)
s7_mid = y + h7 / 2
out.append(
    f'<path d="M {X1} {s7_mid:.1f} H {RAILX} V {S4_MID:.1f} H {X1 + 10}" fill="none" '
    f'stroke="{RAIL}" stroke-width="1.4" stroke-dasharray="5 4"/>'
)
out.append(f'<path d="M {X1 + 10} {S4_MID - 5:.1f} L {X1 + 10} {S4_MID + 5:.1f} L {X1 + 1} {S4_MID:.1f} Z" fill="{RAIL}"/>')
out.append(
    f'<path d="M {RAILX} {S5_MID:.1f} H {X1 + 10}" fill="none" stroke="{RAIL}" stroke-width="1.4" stroke-dasharray="5 4"/>'
)
out.append(f'<path d="M {X1 + 10} {S5_MID - 5:.1f} L {X1 + 10} {S5_MID + 5:.1f} L {X1 + 1} {S5_MID:.1f} Z" fill="{RAIL}"/>')
_rail_mid = (S4_MID + s7_mid) / 2
out.append(
    f'<text x="{RAILX + 14}" y="{_rail_mid:.1f}" font-family="{MONO}" font-size="9.5" fill="{RAIL}" '
    f'text-anchor="middle" transform="rotate(-90 {RAILX + 14} {_rail_mid:.1f})">registry feedback</text>'
)

# ================================================================ footer
fy = y + h7 + 36
hline(X0, fy - 14, X1)
text(X0, fy + 4, "Unchanged by design: shim + GBNF enforcement · single-flight NPU · monitor/registry plumbing · LLM-only policy — stages 2–4 measure, stage 6 judges.", size=10.5, fill=MUTED)
text(X0, fy + 21, "Vibration tokens are input-only; the output vocabulary is the §2.1 axes + severity. Codec and LoRA versions are pinned together — changing the codec invalidates the model.", size=10.5, fill=MUTED)
text(X0, fy + 44, "source: NATIVE_VIBRATION_LLM_PLAN.md §2 §2.1 §3 §6 §8 · 2026-07-07", size=9.5, fill=RAIL, mono=True)

out.append("</svg>")
print("\n".join(out))
