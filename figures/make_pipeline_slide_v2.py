"""Generate QuantAb pipeline overview slide — v2.

Changes from v1:
- REDUCTION section expanded to show three DR method options:
    1. PCA per modality      [Hackathon scope]
    2. PLSR per modality     [Paper scope — label-predictive]
    3. Random Projection     [Null baseline]
- DR section organised as three swim-lane bands; classifiers apply to each.
- Footer updated to reflect multi-DR design.
- Output saved as quantab_pipeline_v2.pptx (v1 unchanged).
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


# ── colours ──────────────────────────────────────────────────────────────────
C_BG        = RGBColor(0x0E, 0x1A, 0x2B)   # dark navy background
C_AB        = RGBColor(0x29, 0x8F, 0xFF)   # antibody track – blue
C_AB_DARK   = RGBColor(0x12, 0x5A, 0xB5)
C_TGT       = RGBColor(0x2E, 0xCC, 0x9A)   # target track – teal
C_TGT_DARK  = RGBColor(0x17, 0x87, 0x63)
C_QM        = RGBColor(0xA0, 0x6A, 0xFF)   # quantum – purple
C_QM_DARK   = RGBColor(0x63, 0x2A, 0xC8)
C_CL        = RGBColor(0xFF, 0x8C, 0x2A)   # classical – orange
C_CL_DARK   = RGBColor(0xC8, 0x5A, 0x00)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT     = RGBColor(0xCC, 0xD6, 0xE8)
C_ARROW     = RGBColor(0x88, 0x99, 0xBB)
C_STORE     = RGBColor(0x3A, 0x4A, 0x6B)   # storage box


def add_box(slide, x, y, w, h, text, fill, text_color=C_WHITE,
            font_size=11, bold=False, border_color=None, border_width=Pt(0),
            subtext=None, subtext_size=9):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = text_color

    if subtext:
        from pptx.util import Pt as _Pt
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = subtext
        r2.font.size = _Pt(subtext_size)
        r2.font.color.rgb = C_LIGHT
        r2.font.bold = False

    return shape


def add_arrow(slide, x1, y1, x2, y2):
    """Add a simple connector arrow."""
    from pptx.util import Emu
    import math
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w < 0.01:
        w = 0.01
    shape = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR.STRAIGHT
        Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    shape.line.color.rgb = C_ARROW
    shape.line.width = Pt(1.5)
    return shape


def add_label(slide, x, y, w, h, text, color=C_LIGHT, size=8, bold=False, align=PP_ALIGN.CENTER):
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txb.text_frame
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    return txb


def add_section_label(slide, x, y, w, h, text, color):
    """Vertical section label."""
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(7.5)
    run.font.bold = True
    run.font.color.rgb = color
    return txb


# extra colours for v2
C_PCA   = RGBColor(0x29, 0x8F, 0xFF)   # PCA — same blue as AB (hackathon)
C_PLSR  = RGBColor(0xFF, 0xD7, 0x00)   # PLSR — gold (paper)
C_PLSR_DARK = RGBColor(0x99, 0x80, 0x00)
C_RP    = RGBColor(0x88, 0x99, 0xBB)   # Random proj — grey (null)
C_RP_DARK   = RGBColor(0x44, 0x55, 0x77)
C_BAND1 = RGBColor(0x12, 0x22, 0x3A)   # PCA band bg
C_BAND2 = RGBColor(0x18, 0x28, 0x18)   # PLSR band bg
C_BAND3 = RGBColor(0x1A, 0x1A, 0x2E)   # RandProj band bg


def add_badge(slide, x, y, w, h, text, fill, text_color=C_WHITE, size=7.5):
    """Small scope/tag badge."""
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = text_color
    return shape


# ── slide setup ───────────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

slide_layout = prs.slide_layouts[6]  # blank
slide = prs.slides.add_slide(slide_layout)

# background
bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = C_BG
bg.line.fill.background()

# ── title ─────────────────────────────────────────────────────────────────────
add_label(slide, 0.2, 0.12, 12.9, 0.45,
          "QuantAb  ·  End-to-End Pipeline  [v2 — DR method comparison]",
          color=C_WHITE, size=20, bold=True)

add_label(slide, 0.2, 0.52, 12.9, 0.28,
          "Three dimensionality-reduction strategies evaluated: "
          "PCA (hackathon) · PLSR (paper, label-predictive) · Random Projection (null baseline) "
          "— same classifiers applied to each",
          color=C_LIGHT, size=9.5)

# ── section dividers ─────────────────────────────────────────────────────────
# columns: INPUT 0.15-1.95 | EMBEDDING 1.95-4.85 | DR FORK 4.85-8.15 | CLASSIFIERS 8.15-13.15

for lx in [1.95, 4.85, 8.15]:
    line = slide.shapes.add_shape(1, Inches(lx), Inches(0.88), Inches(0.01), Inches(6.3))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0x33, 0x44, 0x66)
    line.line.fill.background()

for cx, cw, label in [
    (0.15,  1.75,  "INPUT"),
    (2.0,   2.75,  "EMBEDDING MODEL"),
    (4.9,   3.15,  "DR METHOD  (experimental axis)"),
    (8.2,   4.9,   "CLASSIFIERS  (applied to each DR output)"),
]:
    add_label(slide, cx, 0.88, cw, 0.28, label,
              color=RGBColor(0x66, 0x77, 0x99), size=7.5, bold=True)

# ── swim-lane background bands for the three DR methods ───────────────────────
# Band 1 (PCA / hackathon):   y 1.20 – 2.72
# Band 2 (PLSR / paper):      y 2.80 – 4.32
# Band 3 (RandProj / null):   y 4.40 – 5.92
BAND_H = 1.52
BAND_TOPS = [1.20, 2.80, 4.40]
BAND_COLORS = [C_BAND1, C_BAND2, C_BAND3]

for top, col in zip(BAND_TOPS, BAND_COLORS):
    band = slide.shapes.add_shape(1, Inches(4.87), Inches(top),
                                  Inches(8.26), Inches(BAND_H))
    band.fill.solid()
    band.fill.fore_color.rgb = col
    band.line.fill.background()

# ── track layout ─────────────────────────────────────────────────────────────
# Within each band, AB sub-row is at band_top + 0.38, TGT at band_top + 1.10
AB_OFF  = 0.38   # offset from band top
TGT_OFF = 1.08
ROW_H   = 0.48

# ── TRACK LABELS (left margin) ───────────────────────────────────────────────
AB_Y  = BAND_TOPS[0] + AB_OFF  + ROW_H / 2   # = 1.82
TGT_Y = BAND_TOPS[0] + TGT_OFF + ROW_H / 2   # = 1.82+0.70 = 2.00 ... recalc below
AB_Y  = BAND_TOPS[0] + AB_OFF  + ROW_H / 2
TGT_Y = BAND_TOPS[0] + TGT_OFF + ROW_H / 2

add_label(slide, 0.02, AB_Y  - 0.16, 0.15, 0.35, "AB",  color=C_AB,  size=7.5, bold=True)
add_label(slide, 0.02, TGT_Y - 0.16, 0.15, 0.35, "TGT", color=C_TGT, size=7.5, bold=True)

# ── INPUT boxes ───────────────────────────────────────────────────────────────
add_box(slide, 0.17, AB_Y - ROW_H/2, 1.68, ROW_H,
        "Antibody Seqs", C_AB_DARK, bold=True, font_size=9,
        subtext="VH + VL (paired)",
        border_color=C_AB, border_width=Pt(1.2))

add_box(slide, 0.17, TGT_Y - ROW_H/2, 1.68, ROW_H,
        "Target Protein Seq", C_TGT_DARK, bold=True, font_size=9,
        subtext="Full sequence",
        border_color=C_TGT, border_width=Pt(1.2))

# ── arrows input → model ──────────────────────────────────────────────────────
add_arrow(slide, 1.85, AB_Y,  2.00, AB_Y)
add_arrow(slide, 1.85, TGT_Y, 2.00, TGT_Y)

# ── EMBEDDING MODEL boxes ────────────────────────────────────────────────────
add_box(slide, 2.00, AB_Y - ROW_H/2, 1.52, ROW_H,
        "BALM-paired", C_AB_DARK, bold=True, font_size=10,
        subtext="Paired VH+VL → pooled emb.",
        border_color=C_AB, border_width=Pt(1.5))

add_box(slide, 2.00, TGT_Y - ROW_H/2, 1.52, ROW_H,
        "ESM2", C_TGT_DARK, bold=True, font_size=10,
        subtext="Protein LM → pooled emb.",
        border_color=C_TGT, border_width=Pt(1.5))

# ── arrows model → store ──────────────────────────────────────────────────────
add_arrow(slide, 3.52, AB_Y,  3.68, AB_Y)
add_arrow(slide, 3.52, TGT_Y, 3.68, TGT_Y)

# ── STORE boxes ───────────────────────────────────────────────────────────────
add_box(slide, 3.68, AB_Y  - ROW_H/2, 1.06, ROW_H,
        "Store", C_STORE, font_size=8, subtext="[N, dim_ab]",
        border_color=C_AB, border_width=Pt(1))

add_box(slide, 3.68, TGT_Y - ROW_H/2, 1.06, ROW_H,
        "Store", C_STORE, font_size=8, subtext="[N, dim_tgt]",
        border_color=C_TGT, border_width=Pt(1))

# ── arrows store → DR zone ────────────────────────────────────────────────────
# Both tracks enter at the PCA band (band 1). Then dashed lines split down.
add_arrow(slide, 4.74, AB_Y,  4.90, AB_Y)
add_arrow(slide, 4.74, TGT_Y, 4.90, TGT_Y)

# ── DR METHOD FORK — three bands ──────────────────────────────────────────────
# For each band: scope badge | ab DR box | tgt DR box | arrow out
# AB and TGT rows come in at band 1, then we draw dashed branch lines to 2 & 3.

DR_BOX_X  = 5.15   # x start for DR boxes
DR_BOX_W  = 1.40   # width of each DR box (ab and tgt side by side, small gap)
DR_GAP    = 0.08
DR_OUT_X  = DR_BOX_X + 2 * DR_BOX_W + DR_GAP + 0.10  # x of output arrow start = ~8.13

BADGE_X   = 4.88
BADGE_W   = 0.24

# ── band 1 : PCA (hackathon) ──────────────────────────────────────────────────
B1 = BAND_TOPS[0]
B1_AB  = B1 + AB_OFF
B1_TGT = B1 + TGT_OFF

add_badge(slide, BADGE_X, B1 + 0.55, BADGE_W, 0.38,
          "★\nHACK", fill=RGBColor(0x00, 0x66, 0x33), size=6.5)

add_box(slide, DR_BOX_X, B1_AB, DR_BOX_W, ROW_H,
        "Norm + PCA", C_AB_DARK, font_size=9, bold=True,
        subtext="k_ab dims\n(CV Spearman ρ)",
        border_color=C_AB, border_width=Pt(1.5))

add_box(slide, DR_BOX_X + DR_BOX_W + DR_GAP, B1_TGT, DR_BOX_W, ROW_H,
        "Norm + PCA", C_TGT_DARK, font_size=9, bold=True,
        subtext="k_tgt dims\n(CV Spearman ρ)",
        border_color=C_TGT, border_width=Pt(1.5))

add_label(slide, DR_BOX_X, B1_AB - 0.22, 2 * DR_BOX_W + DR_GAP, 0.22,
          "⚠  normalize each modality independently",
          color=RGBColor(0xFF, 0xCC, 0x44), size=6.5)

# output arrow band 1
add_arrow(slide, DR_BOX_X + DR_BOX_W + DR_GAP, B1_AB + ROW_H/2,
          DR_OUT_X, B1_AB + ROW_H/2)
add_arrow(slide, DR_BOX_X + 2*DR_BOX_W + DR_GAP, B1_TGT + ROW_H/2,
          DR_OUT_X, B1_TGT + ROW_H/2)

# ── band 2 : PLSR (paper) ─────────────────────────────────────────────────────
B2 = BAND_TOPS[1]
B2_AB  = B2 + AB_OFF
B2_TGT = B2 + TGT_OFF

add_badge(slide, BADGE_X, B2 + 0.55, BADGE_W, 0.38,
          "◆\nPAPER", fill=RGBColor(0x77, 0x55, 0x00), size=6.5)

add_box(slide, DR_BOX_X, B2_AB, DR_BOX_W, ROW_H,
        "Norm + PLSR", C_PLSR_DARK, font_size=9, bold=True,
        subtext="k_ab dims\n(label-predictive)",
        border_color=C_PLSR, border_width=Pt(1.5), text_color=C_WHITE)

add_box(slide, DR_BOX_X + DR_BOX_W + DR_GAP, B2_TGT, DR_BOX_W, ROW_H,
        "Norm + PLSR", C_PLSR_DARK, font_size=9, bold=True,
        subtext="k_tgt dims\n(label-predictive)",
        border_color=C_PLSR, border_width=Pt(1.5), text_color=C_WHITE)

add_label(slide, DR_BOX_X, B2_AB - 0.22, 2 * DR_BOX_W + DR_GAP, 0.22,
          "Maximises covariance with affinity  ·  risk: overfits in small-data regime",
          color=C_PLSR, size=6.5)

add_arrow(slide, DR_BOX_X + DR_BOX_W + DR_GAP, B2_AB + ROW_H/2,
          DR_OUT_X, B2_AB + ROW_H/2)
add_arrow(slide, DR_BOX_X + 2*DR_BOX_W + DR_GAP, B2_TGT + ROW_H/2,
          DR_OUT_X, B2_TGT + ROW_H/2)

# ── band 3 : Random Projection (null baseline) ────────────────────────────────
B3 = BAND_TOPS[2]
B3_AB  = B3 + AB_OFF
B3_TGT = B3 + TGT_OFF

add_badge(slide, BADGE_X, B3 + 0.55, BADGE_W, 0.38,
          "∅\nNULL", fill=RGBColor(0x33, 0x33, 0x55), size=6.5)

add_box(slide, DR_BOX_X, B3_AB, DR_BOX_W, ROW_H,
        "Norm + RandProj", C_RP_DARK, font_size=9, bold=True,
        subtext="k_ab dims\n(random linear)",
        border_color=C_RP, border_width=Pt(1.2))

add_box(slide, DR_BOX_X + DR_BOX_W + DR_GAP, B3_TGT, DR_BOX_W, ROW_H,
        "Norm + RandProj", C_RP_DARK, font_size=9, bold=True,
        subtext="k_tgt dims\n(random linear)",
        border_color=C_RP, border_width=Pt(1.2))

add_label(slide, DR_BOX_X, B3_AB - 0.22, 2 * DR_BOX_W + DR_GAP, 0.22,
          "If quantum + RandProj ≈ quantum + PCA → Hilbert space geometry > DR quality",
          color=C_RP, size=6.5)

add_arrow(slide, DR_BOX_X + DR_BOX_W + DR_GAP, B3_AB + ROW_H/2,
          DR_OUT_X, B3_AB + ROW_H/2)
add_arrow(slide, DR_BOX_X + 2*DR_BOX_W + DR_GAP, B3_TGT + ROW_H/2,
          DR_OUT_X, B3_TGT + ROW_H/2)

# branch arrows from input tracks into bands 2 and 3
# (AB and TGT tracks enter at band 1; dashed visual lines show they also feed bands 2 & 3)
for by in [B2_AB + ROW_H/2, B3_AB + ROW_H/2]:
    add_arrow(slide, 4.74, AB_Y, 5.15, by)
for by in [B2_TGT + ROW_H/2, B3_TGT + ROW_H/2]:
    add_arrow(slide, 4.74, TGT_Y, 5.15 + DR_BOX_W + DR_GAP, by)

# ── CLASSIFIERS (right side, three arms, applied to all DR outputs) ────────────
CL_X = 8.18

# Three arms vertically centred within the three bands
CL_Y  = BAND_TOPS[0] + BAND_H / 2        # classical — band 1
QM_Y  = BAND_TOPS[1] + BAND_H / 2        # quantum   — band 2
ABL_Y = BAND_TOPS[2] + BAND_H / 2        # ablation  — band 3

# ── arm: CLASSICAL ────────────────────────────────────────────────────────────
add_label(slide, CL_X, CL_Y - 0.68, 4.85, 0.22,
          "CLASSICAL BASELINE  (×3 DR methods)", color=C_CL, size=7, bold=True)

add_box(slide, CL_X, CL_Y - 0.26, 1.42, 0.52,
        "Concatenate", C_CL_DARK, font_size=9, bold=True,
        subtext="[N, k_ab + k_tgt]",
        border_color=C_CL, border_width=Pt(1.2))

add_arrow(slide, CL_X + 1.42, CL_Y, CL_X + 1.58, CL_Y)

add_box(slide, CL_X + 1.58, CL_Y - 0.26, 1.52, 0.52,
        "SVM · RF · MLP", C_CL_DARK, font_size=9, bold=True,
        subtext="Linear / RBF / Poly",
        border_color=C_CL, border_width=Pt(1.2))

add_arrow(slide, CL_X + 3.10, CL_Y, CL_X + 3.26, CL_Y)

add_box(slide, CL_X + 3.26, CL_Y - 0.26, 1.60, 0.52,
        "Spearman ρ", C_CL, font_size=10, bold=True,
        subtext="vs. affinity (KD)",
        border_color=C_CL_DARK, border_width=Pt(1))

# ── arm: QUANTUM ──────────────────────────────────────────────────────────────
add_label(slide, CL_X, QM_Y - 0.68, 4.85, 0.22,
          "QUANTUM KERNEL  two-register · H_ab ⊗ H_tgt  (×3 DR methods)", color=C_QM, size=7, bold=True)

add_box(slide, CL_X, QM_Y - 0.40, 1.38, 0.36,
        "Register 1", C_QM_DARK, font_size=8, bold=True,
        subtext="k_ab qubits · AngleEnc(ab)",
        border_color=C_AB, border_width=Pt(1))

add_box(slide, CL_X, QM_Y + 0.04, 1.38, 0.36,
        "Register 2", C_QM_DARK, font_size=8, bold=True,
        subtext="k_tgt qubits · AngleEnc(tgt)",
        border_color=C_TGT, border_width=Pt(1))

add_arrow(slide, CL_X + 1.38, QM_Y - 0.22, CL_X + 1.55, QM_Y - 0.22)
add_arrow(slide, CL_X + 1.38, QM_Y + 0.22, CL_X + 1.55, QM_Y + 0.22)

add_box(slide, CL_X + 1.55, QM_Y - 0.35, 1.35, 0.70,
        "Entangling\nCircuit", C_QM_DARK, font_size=8, bold=True,
        subtext="BasicEntangler /\nStronglyEntangling",
        border_color=C_QM, border_width=Pt(1.5))

add_arrow(slide, CL_X + 2.90, QM_Y, CL_X + 3.06, QM_Y)

add_box(slide, CL_X + 3.06, QM_Y - 0.26, 1.20, 0.52,
        "Kernel → SVM", C_QM, font_size=9, bold=True,
        subtext="H_ab ⊗ H_tgt",
        border_color=C_QM_DARK, border_width=Pt(1))

add_arrow(slide, CL_X + 4.26, QM_Y, CL_X + 4.42, QM_Y)
add_box(slide, CL_X + 4.42, QM_Y - 0.26, 0.44, 0.52, "ρ",
        C_QM, font_size=10, bold=True)

add_label(slide, CL_X, QM_Y + 0.42, 4.86, 0.22,
          "Qubit budget: (k_ab, k_tgt) = (6,6)·(8,8)·(10,10)·(14,14)  →  12/16/20/28 qubits total",
          color=RGBColor(0xBB, 0xAA, 0xFF), size=6.5)

# ── arm: ABLATION ─────────────────────────────────────────────────────────────
add_label(slide, CL_X, ABL_Y - 0.64, 4.85, 0.22,
          "ABLATION — single-register on concatenated  (isolates tensor product benefit)",
          color=RGBColor(0x77, 0x88, 0xAA), size=7, bold=True)

add_box(slide, CL_X, ABL_Y - 0.26, 1.42, 0.52,
        "Concatenate", RGBColor(0x2A, 0x3A, 0x55), font_size=9,
        subtext="[N, k_ab + k_tgt]",
        border_color=RGBColor(0x55, 0x66, 0x88), border_width=Pt(1))

add_arrow(slide, CL_X + 1.42, ABL_Y, CL_X + 1.58, ABL_Y)

add_box(slide, CL_X + 1.58, ABL_Y - 0.26, 1.52, 0.52,
        "Single-register QK", RGBColor(0x2A, 0x3A, 0x55), font_size=9,
        subtext="k_ab+k_tgt qubits",
        border_color=RGBColor(0x55, 0x66, 0x88), border_width=Pt(1))

add_arrow(slide, CL_X + 3.10, ABL_Y, CL_X + 3.26, ABL_Y)

add_box(slide, CL_X + 3.26, ABL_Y - 0.26, 1.60, 0.52,
        "Spearman ρ", RGBColor(0x2A, 0x3A, 0x55), font_size=9,
        subtext="tensor product benefit?",
        border_color=RGBColor(0x55, 0x66, 0x88), border_width=Pt(1))

# ── footer ────────────────────────────────────────────────────────────────────
add_label(slide, 0.15, 7.13, 12.9, 0.28,
          "All classifier arms see identical reduced features per DR method  ·  Seed = 42  ·  "
          "Core analysis: learning curves  ·  "
          "DR method is a second experimental axis (PCA vs PLSR vs RandProj)",
          color=RGBColor(0x55, 0x66, 0x88), size=7.5)

# ── save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "quantab_pipeline_v2.pptx"
prs.save(out)
print(f"Saved → {out}")
