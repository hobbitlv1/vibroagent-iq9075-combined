#!/usr/bin/env python3
"""Native-pipeline slide figure in the 'VibroAgent Slide Percent' visual language.

1920x1080 SVG (render at 2x -> 3840x2160), two lanes: CPU sense/tokenize on top,
NPU decide/surface below; cards, storage cylinders, cost pills, timing pills.
"""
from __future__ import annotations

# ------------------------------------------------------------- palette (match reference slide)
INK = "#1E293B"          # titles
BODY = "#64748B"         # body/sub text
SUBTLE = "#94A3B8"       # arrows, faint text
NEUT_FILL = "#F8FAFC"
NEUT_BORD = "#CBD5E1"
TEAL = "#0F766E"         # section labels, dark pills
TEAL_TXT = "#0D9488"
TEAL_FILL = "#F0FDF9"
TEAL_BORD = "#8CE0CE"
TEAL_ICON_BG = "#CCFBEF"
MINT_FILL = "#ECFDF5"
MINT_BORD = "#6EE7B7"
IND = "#4338CA"          # indigo titles
IND_SUB = "#6366F1"
IND_FILL = "#EEF2FF"
IND_BORD = "#A5B4FC"
IND_ICON_BG = "#E0E7FF"
IND_PILL = "#4F46E5"
CYL_FILL = "#E8EDF3"
CYL_TOP = "#CBD5E1"
CYL_BORD = "#94A3B8"
WHITE = "#FFFFFF"

SANS = "DejaVu Sans, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"

W, H = 1920, 1080
out: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s: str, size: float, bold=False, mono=False) -> float:
    k = 0.602 if mono else (0.66 if bold else 0.52)
    return len(s) * size * k


def text(x, y, s, size=13, fill=INK, bold=False, mono=False, anchor="start", spacing=None):
    fam = MONO if mono else SANS
    w = "600" if bold else "400"
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    out.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{w}" '
        f'fill="{fill}" text-anchor="{anchor}"{ls}>{esc(s)}</text>'
    )


def rrect(x, y, w, h, r=14, fill=WHITE, stroke=NEUT_BORD, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def arrow(x0, y0, x1, y1, label=None, dash=None, sw=2.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{SUBTLE}" stroke-width="{sw}"{d}/>')
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    ax, ay = x1, y1
    s = 9
    p1 = (ax - s * math.cos(ang - 0.42), ay - s * math.sin(ang - 0.42))
    p2 = (ax - s * math.cos(ang + 0.42), ay - s * math.sin(ang + 0.42))
    out.append(f'<path d="M {ax:.1f} {ay:.1f} L {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f} Z" fill="{SUBTLE}"/>')
    if label:
        text((x0 + x1) / 2, min(y0, y1) - 8, label, size=11, fill=SUBTLE, anchor="middle")


def elbow(points, dash=None, arrow_end=True):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points)
    out.append(f'<polyline points="{pts}" fill="none" stroke="{SUBTLE}" stroke-width="2"{d}/>')
    if arrow_end:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        arrow(x0, y0, x1, y1, sw=0.001)  # reuse head only; line already drawn


def pill(cx, y, label, fill=TEAL, txt=WHITE, size=13.5, bold=True):
    w = tw(label, size, bold=bold) + 34
    h = 30
    rrect(cx - w / 2, y, w, h, r=15, fill=fill, stroke=fill)
    text(cx, y + 20, label, size=size, fill=txt, bold=bold, anchor="middle")
    return w


def tag_new(x, y, label="new"):
    w = tw(label, 10.5) + 16
    rrect(x - w, y, w, 18, r=9, fill=WHITE, stroke=TEAL_BORD, sw=1.2)
    text(x - w / 2, y + 12.5, label, size=10.5, fill=TEAL, anchor="middle")


def icon_circle(cx, cy, r=21, bg=TEAL_ICON_BG):
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{bg}"/>')


def icon_wave(cx, cy, stroke=TEAL_TXT):
    out.append(f'<polyline points="{cx-13},{cy} {cx-8},{cy-9} {cx-3},{cy+9} {cx+2},{cy-6} {cx+6},{cy+5} {cx+10},{cy-2} {cx+13},{cy}" fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')


def icon_doc(cx, cy, stroke="#475569"):
    out.append(f'<rect x="{cx-9}" y="{cy-12}" width="18" height="24" rx="3" fill="none" stroke="{stroke}" stroke-width="2"/>')
    for dy in (-4, 1, 6):
        out.append(f'<line x1="{cx-4}" y1="{cy+dy}" x2="{cx+4}" y2="{cy+dy}" stroke="{stroke}" stroke-width="2" stroke-linecap="round"/>')


def icon_chat(cx, cy, stroke="#475569"):
    out.append(f'<rect x="{cx-13}" y="{cy-11}" width="26" height="18" rx="6" fill="none" stroke="{stroke}" stroke-width="2.2"/>')
    out.append(f'<path d="M {cx-4} {cy+7} L {cx-1} {cy+13} L {cx+4} {cy+7}" fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linejoin="round"/>')


def icon_mag(cx, cy, stroke=TEAL_TXT):
    out.append(f'<circle cx="{cx-3}" cy="{cy-3}" r="8.5" fill="none" stroke="{stroke}" stroke-width="2.4"/>')
    out.append(f'<line x1="{cx+4}" y1="{cy+4}" x2="{cx+12}" y2="{cy+12}" stroke="{stroke}" stroke-width="2.6" stroke-linecap="round"/>')


def icon_funnel(cx, cy, stroke=TEAL_TXT):
    out.append(f'<path d="M {cx-13} {cy-10} H {cx+13} L {cx+4} {cy+2} V {cy+12} L {cx-4} {cy+8} V {cy+2} Z" fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linejoin="round"/>')


def icon_grid(cx, cy, stroke=TEAL_TXT):
    for dx in (-11, 1):
        for dy in (-11, 1):
            out.append(f'<rect x="{cx+dx}" y="{cy+dy}" width="10" height="10" rx="2" fill="none" stroke="{stroke}" stroke-width="2"/>')


def icon_braces(cx, cy, fill=TEAL_TXT):
    text(cx, cy + 6, "{…}", size=17, fill=fill, bold=True, anchor="middle")


def icon_chip(cx, cy, stroke=IND):
    out.append(f'<rect x="{cx-10}" y="{cy-10}" width="20" height="20" rx="4" fill="none" stroke="{stroke}" stroke-width="2.2"/>')
    out.append(f'<rect x="{cx-4.5}" y="{cy-4.5}" width="9" height="9" rx="1.5" fill="none" stroke="{stroke}" stroke-width="1.8"/>')
    for d in (-6, 0, 6):
        out.append(f'<line x1="{cx+d}" y1="{cy-14}" x2="{cx+d}" y2="{cy-10}" stroke="{stroke}" stroke-width="2"/>')
        out.append(f'<line x1="{cx+d}" y1="{cy+10}" x2="{cx+d}" y2="{cy+14}" stroke="{stroke}" stroke-width="2"/>')


def icon_popup(cx, cy, stroke=IND):
    out.append(f'<rect x="{cx-13}" y="{cy-10}" width="26" height="20" rx="4" fill="none" stroke="{stroke}" stroke-width="2.2"/>')
    out.append(f'<line x1="{cx-13}" y1="{cy-3}" x2="{cx+13}" y2="{cy-3}" stroke="{stroke}" stroke-width="2"/>')
    out.append(f'<circle cx="{cx+9}" cy="{cy-6.7}" r="1.6" fill="{stroke}"/>')


def icon_saved(cx, cy, stroke="#475569"):
    out.append(f'<rect x="{cx-12}" y="{cy-9}" width="21" height="16" rx="3" fill="none" stroke="{stroke}" stroke-width="2.2"/>')
    out.append(f'<circle cx="{cx+10}" cy="{cy-8}" r="4" fill="#E11D48"/>')


def cylinder(x, y, w, h, ry=13):
    out.append(
        f'<path d="M {x} {y+ry} v {h-2*ry} a {w/2} {ry} 0 0 0 {w} 0 v {-(h-2*ry)}" '
        f'fill="{CYL_FILL}" stroke="{CYL_BORD}" stroke-width="1.6"/>'
    )
    out.append(f'<ellipse cx="{x+w/2}" cy="{y+ry}" rx="{w/2}" ry="{ry}" fill="{CYL_TOP}" stroke="{CYL_BORD}" stroke-width="1.6"/>')


def kv(x, y, bold_part, rest, size=12.5, bold_fill=INK, rest_fill=BODY, mono_rest=False):
    text(x, y, bold_part, size=size, fill=bold_fill, bold=True)
    text(x + tw(bold_part, size, bold=True) + 9, y, rest, size=size, fill=rest_fill, mono=mono_rest)


# ================================================================ canvas
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
out.append(f'<rect width="{W}" height="{H}" fill="{WHITE}"/>')

# ---------------------------------------------------------------- title
text(76, 70, "VibroAgent Native Vibration Pipeline", size=34, bold=True)
text(76, 102, "Waveforms become discrete tokens on the CPU; one grammar-locked describe → decide verdict on the NPU, vote-gated into operator popups — every 30 s.",
     size=15, fill=BODY)

# ---------------------------------------------------------------- lane 1 label
text(76, 172, "SENSE & TOKENIZE: EVERY 30 S CYCLE · CPU", size=14.5, fill=TEAL, bold=True, spacing="0.09em")

ROW_Y, CARD_H = 196, 132

def simple_card(x, w, title, sub, tint=None, icon=None, new=False, sub2=None):
    fill = TEAL_FILL if tint == "teal" else NEUT_FILL
    bord = TEAL_BORD if tint == "teal" else NEUT_BORD
    rrect(x, ROW_Y, w, CARD_H, r=14, fill=fill, stroke=bord)
    cx = x + w / 2
    if icon:
        icon_circle(cx, ROW_Y + 34, bg=(TEAL_ICON_BG if tint == "teal" else "#E2E8F0"))
        icon(cx, ROW_Y + 34)
    text(cx, ROW_Y + 82, title, size=15, bold=True, anchor="middle",
         fill=(TEAL if tint == "teal" else INK))
    text(cx, ROW_Y + 103, sub, size=11.5, fill=BODY, anchor="middle")
    if sub2:
        text(cx, ROW_Y + 119, sub2, size=11.5, fill=BODY, anchor="middle")
    if new:
        tag_new(x + w - 8, ROW_Y + 8)
    return x + w

# n1 boards
x1e = simple_card(76, 208, "Six STWIN.box boards", "IIS3DWB · 1 ref + 5 targets", icon=icon_wave)
arrow(x1e + 6, ROW_Y + CARD_H / 2, x1e + 38, ROW_Y + CARD_H / 2)
# n2 ring files (cylinder)
c2x, c2w = x1e + 44, 176
cylinder(c2x, ROW_Y + 2, c2w, CARD_H - 4)
icon_circle(c2x + c2w / 2, ROW_Y + 40, bg="#F1F5F9"); icon_doc(c2x + c2w / 2, ROW_Y + 40)
text(c2x + c2w / 2, ROW_Y + 84, ".dat ring files", size=15, bold=True, anchor="middle")
text(c2x + c2w / 2, ROW_Y + 104, "Logger · USB · per board", size=11.5, fill=BODY, anchor="middle")
arrow(c2x + c2w + 6, ROW_Y + CARD_H / 2, c2x + c2w + 38, ROW_Y + CARD_H / 2)
# n3 chat guard
n3x = c2x + c2w + 44
x3e = simple_card(n3x, 198, "Chat using the NPU?", "yes → skip · retry in 30 s", icon=icon_chat)
arrow(x3e + 6, ROW_Y + CARD_H / 2, x3e + 40, ROW_Y + CARD_H / 2, label="no")
# n4 read window
n4x = x3e + 46
x4e = simple_card(n4x, 198, "Read latest window", "10 s live accel · every board", tint="teal", icon=icon_mag)
arrow(x4e + 6, ROW_Y + CARD_H / 2, x4e + 38, ROW_Y + CARD_H / 2)
# n5 decimate
n5x = x4e + 44
x5e = simple_card(n5x, 196, "Decimate + normalize", "26.7 kHz → 400 Hz · anti-alias", tint="teal", icon=icon_funnel, new=True,
                  sub2="level exits as one scalar (rms g)")
arrow(x5e + 6, ROW_Y + CARD_H / 2, x5e + 38, ROW_Y + CARD_H / 2)

# n6 codec big card
n6x, n6w, n6h = x5e + 44, 300, 252
n6y = ROW_Y - 6
rrect(n6x, n6y, n6w, n6h, r=16, fill=TEAL_FILL, stroke=TEAL_BORD)
icon_circle(n6x + n6w / 2, n6y + 32, bg=TEAL_ICON_BG); icon_grid(n6x + n6w / 2, n6y + 32)
text(n6x + n6w / 2, n6y + 74, "VQ codec — “the eye” · CPU", size=15.5, bold=True, fill=TEAL, anchor="middle")
ky = n6y + 100
for b, r in [("Encode", "— 1-D conv, tri-axis"),
             ("Quantize", "— RVQ 2 × 1024, frozen v1"),
             ("Codes", "— 250 per sensor-window"),
             ("codec_fit", "— novelty scalar out")]:
    kv(n6x + 22, ky, b, r, size=12.5)
    ky += 22
out.append(f'<line x1="{n6x+22}" y1="{ky-6}" x2="{n6x+n6w-22}" y2="{ky-6}" stroke="{TEAL_BORD}" stroke-width="1"/>')
text(n6x + n6w / 2, ky + 14, "Input: 10 s window × 6 boards", size=11.5, fill=BODY, anchor="middle")
tag_new(n6x + n6w - 8, n6y + 8)
arrow(n6x + n6w + 6, ROW_Y + CARD_H / 2, n6x + n6w + 38, ROW_Y + CARD_H / 2)

# n7 native prompt
n7x = n6x + n6w + 44
n7w = 1844 - n7x
rrect(n7x, ROW_Y, n7w, CARD_H, r=14, fill=TEAL_FILL, stroke=TEAL_BORD)
icon_circle(n7x + n7w / 2, ROW_Y + 34, bg=TEAL_ICON_BG); icon_braces(n7x + n7w / 2, ROW_Y + 34)
text(n7x + n7w / 2, ROW_Y + 82, "Native token prompt", size=15, bold=True, fill=TEAL, anchor="middle")
text(n7x + n7w / 2, ROW_Y + 103, "≈ 2.7–3.8 k tok of 6144:", size=11.5, fill=BODY, anchor="middle")
text(n7x + n7w / 2, ROW_Y + 119, "codes ×6 · context · exemplar", size=11.5, fill=BODY, anchor="middle")
tag_new(n7x + n7w - 8, ROW_Y + 8)

# --------- storage row under codec/prompt
sy = 480
cylA_x, cylA_w = n6x + 30, 240
cylinder(cylA_x, sy, cylA_w, 100)
text(cylA_x + cylA_w / 2, sy + 46, "Codebooks v1 — frozen", size=13.5, bold=True, anchor="middle")
text(cylA_x + cylA_w / 2, sy + 66, "the token dictionary", size=11.5, fill=BODY, anchor="middle")
arrow(cylA_x + cylA_w / 2, sy - 4, cylA_x + cylA_w / 2, n6y + n6h + 10, sw=2)
# registry cylinder
cylB_x, cylB_w = 1844 - 240, 240
cylinder(cylB_x, sy, cylB_w, 100)
text(cylB_x + cylB_w / 2, sy + 40, "Anomaly registry", size=13.5, bold=True, anchor="middle")
text(cylB_x + cylB_w / 2, sy + 60, "confirmed exemplars → few-shot", size=11.5, fill=BODY, anchor="middle")
text(cylB_x + cylB_w / 2, sy + 78, "≈ 300 tok · one encounter per fleet", size=11.5, fill=BODY, anchor="middle")
arrow(cylB_x + cylB_w / 2, sy - 4, cylB_x + cylB_w / 2, ROW_Y + CARD_H + 10, sw=2)

# --------- brace + cost pills lane 1
by = ROW_Y + CARD_H + 26
out.append(
    f'<path d="M 84 {by} Q 84 {by+16} 104 {by+16} H {(84+x4e)/2 - 14} Q {(84+x4e)/2} {by+16} {(84+x4e)/2} {by+30} '
    f'Q {(84+x4e)/2} {by+16} {(84+x4e)/2 + 14} {by+16} H {x4e-20} Q {x4e} {by+16} {x4e} {by}" '
    f'fill="none" stroke="{TEAL}" stroke-width="2"/>'
)
pill((84 + x4e) / 2, by + 34, "< 2 % · guard + window read · CPU")
pill(1300, 594, "≈ 3 % · decimate + codec + prompt · CPU")

# ---------------------------------------------------------------- elbow lane1 -> lane2
out.append(f'<polyline points="1844,{ROW_Y + 40} 1876,{ROW_Y + 40} 1876,630 640,630 640,658" fill="none" stroke="{SUBTLE}" stroke-width="2"/>')
out.append(f'<path d="M 640 666 L 634.5 656 L 645.5 656 Z" fill="{SUBTLE}"/>')

# ---------------------------------------------------------------- lane 2
text(76, 646, "DECIDE & SURFACE: ON THE NPU · NATIVE READING", size=14.5, fill=TEAL, bold=True, spacing="0.09em")

L2Y = 664
# big NPU card
npx, npw, nph = 76, 930, 300
rrect(npx, L2Y, npw, nph, r=16, fill=IND_FILL, stroke=IND_BORD)
icon_circle(npx + 44, L2Y + 44, bg=IND_ICON_BG); icon_chip(npx + 44, L2Y + 44)
text(npx + 84, L2Y + 38, "Qwen3-4B + vibration LoRA — greedy · temp 0", size=17, bold=True, fill=IND)
text(npx + 84, L2Y + 60, "GenieX shim :18181 · llama.cpp · Hexagon NPU · n_ctx 6144", size=12.5, fill=IND_SUB)
text(npx + 84, L2Y + 78, "GBNF grammar forces describe → decide JSON · vibration tokens input-only", size=12.5, fill=IND_SUB)
tag_new(npx + npw - 10, L2Y + 10, label="new weights · merged, Q4_0")
# sub-card 1
s1x, s1y, s1w, s1h = npx + 26, L2Y + 100, 428, 180
rrect(s1x, s1y, s1w, s1h, r=12, fill=WHITE, stroke="#C7D2FE")
text(s1x + 18, s1y + 30, "1 – Native verdict (call 1)", size=14.5, bold=True, fill=IND)
pw = tw("≈ 70 %", 12.5, bold=True) + 26
rrect(s1x + s1w - pw - 14, s1y + 12, pw, 24, r=12, fill=IND_PILL, stroke=IND_PILL)
text(s1x + s1w - pw / 2 - 14, s1y + 28.5, "≈ 70 %", size=12.5, fill=WHITE, bold=True, anchor="middle")
yy = s1y + 60
for b, r in [("Reads", "— ref + target code streams"),
             ("Describes", "— amplitude, spectral character,"),
             ("", "impulsiveness, persistence"),
             ("Decides", "— severity + affected sensors"),
             ("Plus", "— confidence + duration check")]:
    if b:
        kv(s1x + 18, yy, b, r, size=12.5)
    else:
        text(s1x + 18 + tw("Describes", 12.5, bold=True) + 8, yy, r, size=12.5, fill=BODY)
    yy += 24
# sub-card 2
s2x, s2y, s2w, s2h = s1x + s1w + 22, L2Y + 100, npw - s1w - 26 * 2 - 22, 180
rrect(s2x, s2y, s2w, s2h, r=12, fill=WHITE, stroke="#C7D2FE")
text(s2x + 18, s2y + 30, "2 – Vote on deviation", size=14.5, bold=True, fill=IND)
pw = tw("≈ 25 %", 12.5, bold=True) + 26
rrect(s2x + s2w - pw - 14, s2y + 12, pw, 24, r=12, fill=IND_PILL, stroke=IND_PILL)
text(s2x + s2w - pw / 2 - 14, s2y + 28.5, "≈ 25 %", size=12.5, fill=WHITE, bold=True, anchor="middle")
yy = s2y + 60
for b, r in [("Trigger", "— any non-normal severity"),
             ("Vote", "— k = 5 seeded re-runs"),
             ("Confidence", "— vote share per axis"),
             ("Check", "— adversarial self-check call"),
             ("Explain", "— call 2 prose, never parsed")]:
    kv(s2x + 18, yy, b, r, size=12.5)
    yy += 24

arrow(npx + npw + 8, L2Y + nph / 2, npx + npw + 42, L2Y + nph / 2)

# popup builder card
ppx, ppy, ppw, pph = npx + npw + 48, L2Y + 28, 330, 232
rrect(ppx, ppy, ppw, pph, r=14, fill=IND_FILL, stroke=IND_BORD)
icon_circle(ppx + ppw / 2, ppy + 36, bg=IND_ICON_BG); icon_popup(ppx + ppw / 2, ppy + 36)
text(ppx + ppw / 2, ppy + 80, "Popup builder", size=15.5, bold=True, fill=IND, anchor="middle")
yy = ppy + 108
for b, r in [("Builds", "— severity + message"),
             ("From", "— axes, vote share, history"),
             ("Sends", "— a desktop notification")]:
    kv(ppx + 20, yy, b, r, size=12.5, bold_fill=IND)
    yy += 24
text(ppx + ppw / 2, yy + 8, "shows vote share when uncertain", size=11.5, fill=BODY, anchor="middle")
pill(ppx + ppw / 2, ppy + pph + 14, "< 1 %", fill=IND_PILL)

arrow(ppx + ppw + 8, L2Y + nph / 2, ppx + ppw + 42, L2Y + nph / 2)

# anomaly episode cylinder
aex, aey, aew, aeh = ppx + ppw + 48, L2Y - 6, 1844 - (ppx + ppw + 48), 306
cylinder(aex, aey, aew, aeh, ry=16)
icon_circle(aex + aew / 2, aey + 48, bg="#F1F5F9"); icon_saved(aex + aew / 2, aey + 48)
text(aex + aew / 2, aey + 92, "Anomaly episode saved", size=15.5, bold=True, anchor="middle")
yy = aey + 122
for b, r, mono_rest in [("Popup", "— graph page, severity tint", False),
                        ("Log", "— anomaly_windows.jsonl", True),
                        ("Codes", "— saved with the episode", False),
                        ("Panel", "— episodes, revisit any time", False)]:
    kv(aex + 30, yy, b, r, size=12.5, mono_rest=mono_rest)
    yy += 24
out.append(f'<line x1="{aex+30}" y1="{yy-6}" x2="{aex+aew-30}" y2="{yy-6}" stroke="{CYL_BORD}" stroke-width="1"/>')
kv(aex + 30, yy + 16, "Quiet", "— “Agent Normal”, not saved", size=12.5)
pill(aex + aew / 2, aey + aeh + 14, "< 1 %", fill=IND_PILL)

# dashed return: episode top -> registry bottom (learning loop)
aeCX = aex + aew / 2
regCX = cylB_x + cylB_w / 2
out.append(
    f'<polyline points="{aeCX:.1f},{aey - 8:.1f} {aeCX:.1f},620 {regCX:.1f},620 {regCX:.1f},590" '
    f'fill="none" stroke="{SUBTLE}" stroke-width="2" stroke-dasharray="6 5"/>'
)
out.append(f'<path d="M {regCX - 5.5:.1f} 592 L {regCX + 5.5:.1f} 592 L {regCX:.1f} 582 Z" fill="{SUBTLE}"/>')
text((aeCX + regCX) / 2, 612, "confirmed → exemplars + LoRA refresh", size=10.5, fill=SUBTLE, anchor="middle")

# ---------------------------------------------------------------- bottom pills
def bottom_pill(cx, label, indigo=False):
    size = 13.5
    w = tw(label, size, bold=True) + 44
    h = 40
    fill = IND_FILL if indigo else MINT_FILL
    bord = IND_BORD if indigo else MINT_BORD
    col = IND if indigo else TEAL
    rrect(cx - w / 2, 1014, w, h, r=12, fill=fill, stroke=bord)
    text(cx, 1039, label, size=size, fill=col, bold=True, anchor="middle")

bottom_pill(320, "≈ 25–35 s · steady native check (R6: measure)")
bottom_pill(800, "escalation adds vote_k = 5 · on deviation only", indigo=True)
bottom_pill(1230, "0 s when chatting · skipped")

text(1844, 1070, "Target: GenieX shim :18181 · Hexagon NPU · n_ctx 6144 · NATIVE_VIBRATION_LLM_PLAN.md §2–§2.1 · 2026-07-07",
     size=12.5, fill=SUBTLE, anchor="end")

out.append("</svg>")
print("\n".join(out))
