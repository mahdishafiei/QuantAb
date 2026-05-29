"""Generate QuantAb pipeline overview slide."""

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
          "QuantAb  ·  End-to-End Pipeline",
          color=C_WHITE, size=20, bold=True)

add_label(slide, 0.2, 0.52, 12.9, 0.28,
          "Paired antibody (VH+VL) and target protein embeddings → "
          "separate dimensionality reduction → classical & quantum classifiers",
          color=C_LIGHT, size=9.5)

# ── section dividers (thin vertical lines) ────────────────────────────────────
#   columns: 0.2 | [INPUT 0.2-2.0] | 2.0 | [EMBEDDING 2.0-5.0] |
#             5.0 | [REDUCTION 5.0-7.4] | 7.4 | [CLASSIFIERS 7.4-13.1]

for lx in [2.05, 5.05, 7.45]:
    line = slide.shapes.add_shape(1, Inches(lx), Inches(0.88), Inches(0.01), Inches(6.45))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0x33, 0x44, 0x66)
    line.line.fill.background()

# column headers
for cx, cw, label in [
    (0.2,  1.8,  "INPUT"),
    (2.1,  2.85, "EMBEDDING MODEL"),
    (5.1,  2.25, "REDUCTION"),
    (7.5,  5.6,  "CLASSIFIERS"),
]:
    add_label(slide, cx, 0.88, cw, 0.28, label,
              color=RGBColor(0x66, 0x77, 0x99), size=7.5, bold=True)

# ── track y centres ───────────────────────────────────────────────────────────
AB_Y  = 2.1   # antibody track centre
TGT_Y = 4.7   # target track centre

ROW_H = 0.62  # box height
ROW_W = 1.55  # standard box width

# ── TRACK LABELS ─────────────────────────────────────────────────────────────
add_label(slide, 0.05, AB_Y - 0.18,  0.15, ROW_H + 0.36, "AB",
          color=C_AB, size=8, bold=True)
add_label(slide, 0.05, TGT_Y - 0.18, 0.15, ROW_H + 0.36, "TGT",
          color=C_TGT, size=8, bold=True)

# ── INPUT boxes ───────────────────────────────────────────────────────────────
add_box(slide, 0.22, AB_Y - ROW_H/2, 1.7, ROW_H,
        "Antibody Sequences", C_AB_DARK, bold=True, font_size=10,
        subtext="Heavy chain (VH)\n+ Light chain (VL)",
        border_color=C_AB, border_width=Pt(1.2))

add_box(slide, 0.22, TGT_Y - ROW_H/2, 1.7, ROW_H,
        "Target Protein Seq", C_TGT_DARK, bold=True, font_size=10,
        subtext="Full protein sequence",
        border_color=C_TGT, border_width=Pt(1.2))

# ── arrows input → model ──────────────────────────────────────────────────────
add_arrow(slide, 1.92, AB_Y,  2.12, AB_Y)
add_arrow(slide, 1.92, TGT_Y, 2.12, TGT_Y)

# ── EMBEDDING MODEL boxes ─────────────────────────────────────────────────────
add_box(slide, 2.12, AB_Y - ROW_H/2, 1.72, ROW_H,
        "BALM-paired", C_AB_DARK, bold=True, font_size=11,
        subtext="Paired VH+VL input\nPooled embedding",
        border_color=C_AB, border_width=Pt(1.5))

add_box(slide, 2.12, TGT_Y - ROW_H/2, 1.72, ROW_H,
        "ESM2", C_TGT_DARK, bold=True, font_size=11,
        subtext="Protein LM\nPooled embedding",
        border_color=C_TGT, border_width=Pt(1.5))

# ── arrows model → store ──────────────────────────────────────────────────────
add_arrow(slide, 3.84, AB_Y,  4.05, AB_Y)
add_arrow(slide, 3.84, TGT_Y, 4.05, TGT_Y)

# ── STORAGE boxes ─────────────────────────────────────────────────────────────
add_box(slide, 4.05, AB_Y - ROW_H/2, 0.9, ROW_H,
        "Store", C_STORE, font_size=9,
        subtext="[N, dim_ab]",
        border_color=C_AB, border_width=Pt(1))

add_box(slide, 4.05, TGT_Y - ROW_H/2, 0.9, ROW_H,
        "Store", C_STORE, font_size=9,
        subtext="[N, dim_tgt]",
        border_color=C_TGT, border_width=Pt(1))

# ── arrows store → reduction ──────────────────────────────────────────────────
add_arrow(slide, 4.95, AB_Y,  5.12, AB_Y)
add_arrow(slide, 4.95, TGT_Y, 5.12, TGT_Y)

# ── REDUCTION boxes ───────────────────────────────────────────────────────────
add_box(slide, 5.12, AB_Y - ROW_H/2, 2.18, ROW_H,
        "Normalize + PCA", C_AB_DARK, font_size=10, bold=True,
        subtext="z-score per dim → k_ab components\n(k set by CV Spearman ρ)",
        border_color=C_AB, border_width=Pt(1.2))

add_box(slide, 5.12, TGT_Y - ROW_H/2, 2.18, ROW_H,
        "Normalize + PCA", C_TGT_DARK, font_size=10, bold=True,
        subtext="z-score per dim → k_tgt components\n(k set by CV Spearman ρ)",
        border_color=C_TGT, border_width=Pt(1.2))

# note: normalize independently — do NOT do joint PCA
add_label(slide, 5.08, 3.45, 2.28, 0.28,
          "⚠  Normalize each modality independently",
          color=RGBColor(0xFF, 0xCC, 0x44), size=7.5)

# ── arrows reduction → classifiers ────────────────────────────────────────────
# antibody → both arms
add_arrow(slide, 7.30, AB_Y,  7.52, AB_Y)
# target → both arms
add_arrow(slide, 7.30, TGT_Y, 7.52, TGT_Y)

# ── CLASSIFIER section ────────────────────────────────────────────────────────
# Split into two sub-columns: classical (top) and quantum (bottom)
# with the two tracks converging into each

CL_Y  = 1.82   # classical arm centre
QM_Y  = 4.18   # quantum arm centre
ABL_Y = 5.90   # ablation arm centre

ARM_W = 5.50

# convergence lines (ab & tgt → classical arm)
add_arrow(slide, 7.52, AB_Y,  7.72, CL_Y)
add_arrow(slide, 7.52, TGT_Y, 7.72, CL_Y)

# convergence lines (ab & tgt → quantum arm)
add_arrow(slide, 7.52, AB_Y,  7.72, QM_Y)
add_arrow(slide, 7.52, TGT_Y, 7.72, QM_Y)

# convergence lines (ab & tgt → ablation arm)
add_arrow(slide, 7.52, AB_Y,  7.72, ABL_Y)
add_arrow(slide, 7.52, TGT_Y, 7.72, ABL_Y)

# ── CLASSICAL ARM ─────────────────────────────────────────────────────────────
CL_X = 7.72

add_box(slide, CL_X, CL_Y - 0.29, 1.52, 0.58,
        "Concatenate", C_CL_DARK, font_size=9, bold=True,
        subtext="[N, k_ab + k_tgt]",
        border_color=C_CL, border_width=Pt(1.2))

add_arrow(slide, CL_X + 1.52, CL_Y, CL_X + 1.72, CL_Y)

add_box(slide, CL_X + 1.72, CL_Y - 0.29, 1.62, 0.58,
        "SVM  ·  RF  ·  MLP", C_CL_DARK, font_size=9, bold=True,
        subtext="Linear / RBF / Poly kernels",
        border_color=C_CL, border_width=Pt(1.2))

add_arrow(slide, CL_X + 3.34, CL_Y, CL_X + 3.54, CL_Y)

add_box(slide, CL_X + 3.54, CL_Y - 0.29, 1.72, 0.58,
        "Spearman ρ", C_CL, font_size=10, bold=True,
        subtext="vs. affinity (KD)",
        border_color=C_CL_DARK, border_width=Pt(1))

add_label(slide, CL_X - 0.05, CL_Y - 0.60, 5.45, 0.24,
          "CLASSICAL BASELINE", color=C_CL, size=7.5, bold=True)

# ── QUANTUM ARM ───────────────────────────────────────────────────────────────
QX = 7.72

# register 1
add_box(slide, QX, QM_Y - 0.58, 1.68, 0.50,
        "Register 1", C_QM_DARK, font_size=9, bold=True,
        subtext="k_ab qubits  ·  AngleEncoding(ab)",
        border_color=C_AB, border_width=Pt(1))

# register 2
add_box(slide, QX, QM_Y + 0.08, 1.68, 0.50,
        "Register 2", C_QM_DARK, font_size=9, bold=True,
        subtext="k_tgt qubits  ·  AngleEncoding(tgt)",
        border_color=C_TGT, border_width=Pt(1))

add_arrow(slide, QX + 1.68, QM_Y - 0.33, QX + 1.88, QM_Y - 0.33)
add_arrow(slide, QX + 1.68, QM_Y + 0.33, QX + 1.88, QM_Y + 0.33)

add_box(slide, QX + 1.88, QM_Y - 0.43, 1.52, 0.86,
        "Entangling\nCircuit", C_QM_DARK, font_size=9, bold=True,
        subtext="BasicEntangler\nor StronglyEntangling\n(2 ansatz variants)",
        border_color=C_QM, border_width=Pt(1.5))

add_arrow(slide, QX + 3.40, QM_Y, QX + 3.60, QM_Y)

add_box(slide, QX + 3.60, QM_Y - 0.29, 1.30, 0.58,
        "Kernel\nMatrix → SVM", C_QM, font_size=9, bold=True,
        subtext="H_ab ⊗ H_tgt",
        border_color=C_QM_DARK, border_width=Pt(1))

add_arrow(slide, QX + 4.90, QM_Y, QX + 5.10, QM_Y)

add_box(slide, QX + 5.10, QM_Y - 0.29, 0.18, 0.58,    # just endpoint
        "ρ", C_QM, font_size=10, bold=True)

add_label(slide, QX - 0.05, QM_Y - 0.78, 5.45, 0.24,
          "QUANTUM KERNEL  (two-register, tensor product Hilbert space)", color=C_QM, size=7.5, bold=True)

# qubit budget note
add_label(slide, QX + 1.86, QM_Y + 0.50, 3.60, 0.30,
          "Qubit budget: (k_ab, k_tgt) = (6,6) / (8,8) / (10,10) / (14,14)  →  12 / 16 / 20 / 28 total",
          color=RGBColor(0xBB, 0xAA, 0xFF), size=7.5)

# ── ABLATION ARM ──────────────────────────────────────────────────────────────
AX = 7.72

add_box(slide, AX, ABL_Y - 0.26, 1.52, 0.52,
        "Concatenate", RGBColor(0x2A, 0x3A, 0x55), font_size=9,
        subtext="[N, k_ab + k_tgt]",
        border_color=RGBColor(0x55, 0x66, 0x88), border_width=Pt(1))

add_arrow(slide, AX + 1.52, ABL_Y, AX + 1.72, ABL_Y)

add_box(slide, AX + 1.72, ABL_Y - 0.26, 1.52, 0.52,
        "Single-register\nQuantum Kernel", RGBColor(0x2A, 0x3A, 0x55), font_size=9,
        subtext="k_ab + k_tgt qubits",
        border_color=RGBColor(0x55, 0x66, 0x88), border_width=Pt(1))

add_arrow(slide, AX + 3.24, ABL_Y, AX + 3.44, ABL_Y)

add_box(slide, AX + 3.44, ABL_Y - 0.26, 1.86, 0.52,
        "Spearman ρ", RGBColor(0x2A, 0x3A, 0x55), font_size=9,
        subtext="Isolates tensor product benefit",
        border_color=RGBColor(0x55, 0x66, 0x88), border_width=Pt(1))

add_label(slide, AX - 0.05, ABL_Y - 0.50, 5.45, 0.24,
          "ABLATION — single-register on concatenated (controls for tensor product structure)",
          color=RGBColor(0x77, 0x88, 0xAA), size=7.5, bold=True)

# ── legend / footer ───────────────────────────────────────────────────────────
add_label(slide, 0.2, 7.12, 12.9, 0.28,
          "All arms see identical reduced features  ·  Same PCA projection  ·  Seed = 42  ·  "
          "Primary metric: Spearman ρ(predicted, measured KD)  ·  Core analysis: learning curves",
          color=RGBColor(0x55, 0x66, 0x88), size=7.5)

# ── save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "quantab_pipeline.pptx"
prs.save(out)
print(f"Saved → {out}")
