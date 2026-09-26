#!/usr/bin/env python3
"""
Figure -- INS-HDGS-CMT preprocessing pipeline: builds a .drawio (mxGraph) source describing every
step from raw EEG / eye-tracking streams to the tensors and graphs that feed the architecture in
fig2_Architecture.drawio. Every step, parameter and ordering below was verified against the actual
Phase 2-4 pipeline code, not just the manuscript prose (see docs/REVISION_RUNS.md for the source
audit). Three things this figure deliberately gets right, to stay consistent with the corrected
Figure 2:

  1. Band-power features feed the dynamic-graph construction AND the LIF spiking encoder as two
     independent, parallel branches (matching Fig. 2 panel A) -- the graph does not feed the LIF
     encoder.
  2. Channel selection and the ROI dwell vector are two independent, parallel outputs of the ET
     epoching step (matching Fig. 2 panel B / the dashed ROI-modulation link), not a sequential
     chain.
  3. Channel harmonization (zero-fill the 4 channels S01 is missing, back to the full 24-wide
     montage, then drop all 5 non-cortical channels to reach the canonical 19) happens AFTER
     signal preprocessing (CAR, filtering, ICA), not before -- CAR and ICA run on each subject's
     own post-bad-channel-removal channel set. This is more precise than the manuscript's Sec. 2.2
     prose, which telescopes the Phase-2 preprocessing step and the later Phase-8 model-loading
     harmonizer into one sentence.

Layout note: the two modalities are cleaned independently (row 1), converge once into a single
"temporal alignment" step (row 2, avoids any misleading direct line between differently-tall
lanes), then diverge again into per-modality epoching / feature construction (row 3) before
feeding the architecture (row 4). The engagement_phase3d label computation is drawn as a
standalone, arrow-free panel (row 5) to avoid implying it sits inline in the model's forward pass.

Output: paper/figures/fig_preprocessing.drawio -- open at https://app.diagrams.net to inspect or
edit further. Render with scripts/figures/render_drawio.py.

Run:  python scripts/figures/fig_preprocessing.py
"""
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "fig_preprocessing.drawio"

PAL = {
    "eeg": dict(head="#2E5AAC", fill="#EAF0FB"),
    "et":  dict(head="#3C7A2E", fill="#EEF6EA"),
    "aln": dict(head="#6A4C93", fill="#F1ECF8"),
    "lbl": dict(head="#5C6670", fill="#EEF0F2"),
    "out": dict(head="#B23A48", fill="#FBE7E9"),
}

cells = []
_id = [1]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def vertex(x, y, w, h, value="", style="", parent="1", cid=None):
    cid = cid or nid()
    cells.append(f'<mxCell id="{cid}" value="{escape(value)}" style="{style}" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def edge(src, tgt, style="", parent="1", label=""):
    cid = nid()
    cells.append(f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" edge="1" parent="{parent}" '
                 f'source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    return cid


def swimlane(x, y, w, h, title, pal):
    style = (f"swimlane;rounded=1;arcSize=4;fillColor={pal['fill']};strokeColor={pal['head']};strokeWidth=2;"
             f"startSize=34;fontColor=#FFFFFF;fontStyle=1;fontSize=14;swimlaneFillColor={pal['fill']};"
             f"swimlaneLine=0;")
    cid = nid()
    cells.append(f'<mxCell id="{cid}" value="{escape(title)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    bar = nid()
    cells.append(f'<mxCell id="{bar}" value="" style="rounded=0;fillColor={pal["head"]};strokeColor=none;" '
                 f'vertex="1" parent="{cid}"><mxGeometry x="0" y="0" width="{w}" height="34" as="geometry"/></mxCell>')
    lbl = nid()
    cells.append(f'<mxCell id="{lbl}" value="{escape(title)}" style="text;html=1;fontColor=#FFFFFF;fontStyle=1;'
                 f'fontSize=14;align=center;verticalAlign=middle;" vertex="1" parent="{cid}">'
                 f'<mxGeometry x="0" y="0" width="{w}" height="34" as="geometry"/></mxCell>')
    return cid


def step(parent, x, y, w, h, num, title, sub, pal, fill="#FFFFFF"):
    box = nid()
    cells.append(f'<mxCell id="{box}" value="&lt;b&gt;{escape(num)}. {escape(title)}&lt;/b&gt;&lt;br&gt;'
                 f'&lt;span style=&quot;font-size:10.5px;color:#444&quot;&gt;{escape(sub)}&lt;/span&gt;" '
                 f'style="rounded=1;arcSize=8;fillColor={fill};strokeColor=#BBBBBB;align=left;'
                 f'verticalAlign=middle;spacingLeft=8;spacingRight=6;fontColor={pal["head"]};fontSize=12.5;'
                 f'html=1;whiteSpace=wrap;" vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" '
                 f'height="{h}" as="geometry"/></mxCell>')
    return box


ARROW = "edgeStyle=none;rounded=0;html=1;strokeColor=#3A3A3A;strokeWidth=2;endArrow=block;endFill=1;"
EEG_ARROW = "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#2E5AAC;strokeWidth=2;endArrow=block;endFill=1;rounded=0;"
ET_ARROW = "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#3C7A2E;strokeWidth=2;endArrow=block;endFill=1;rounded=0;"

# ---------------------------------------------------------------------------
title_id = nid()
cells.append(f'<mxCell id="{title_id}" value="INS-HDGS-CMT Preprocessing Pipeline" '
             f'style="text;html=1;align=center;fontStyle=1;fontSize=24;" vertex="1" parent="1">'
             f'<mxGeometry x="20" y="10" width="1260" height="40" as="geometry"/></mxCell>')

CW, GAP, SH = 578, 12, 66

# ============================= Row 1: independent signal cleaning =============================
p, q = PAL["eeg"], PAL["et"]
A1 = swimlane(20, 60, 610, 518, "EEG Signal Cleaning", p)
eeg_clean = [
    ("1", "Raw EEG", "24-channel montage (10-20 layout), native 300 Hz, per-subject LSL stream"),
    ("2", "Bad-channel handling", "S01 only: 4 non-cortical channels dropped (X1, X2, X3, TRG); "
     "no scalp electrode removed for any subject"),
    ("3", "Common-average reference", "subtract per-timepoint mean across each subject's remaining "
     "channel set (runs before harmonization -- see note)"),
    ("4", "Band-pass filter", "Butterworth order 4, filtfilt, 1-45 Hz"),
    ("5", "Notch filter", "iirnotch + filtfilt, 50 Hz, Q = 30"),
    ("6", "ICA artifact removal", "FastICA, 15 components, seed 42, max_iter 1000"),
]
a1_ids = []
for i, (num, t, s) in enumerate(eeg_clean):
    a1_ids.append(step(A1, 16, 46 + i * (SH + GAP), CW, SH, num, t, s, p))
for i in range(len(a1_ids) - 1):
    edge(a1_ids[i], a1_ids[i + 1], ARROW, parent=A1)

B1 = swimlane(680, 60, 610, 518, "Eye-Tracking Signal Cleaning", q)
et_clean = [
    ("1", "Raw eye-tracking", "6-channel binocular stream (L/R gaze x, y + pupil), native 120 Hz"),
    ("2", "Blink / dropout gap-fill", "linear interpolation, max gap 12 samples (~100 ms)"),
    ("3", "Smoothing", "centered rolling mean, window = 5"),
    ("4", "Fixation detection", "velocity threshold 80 px/s, min duration 12 samples, binocular fusion"),
]
b1_ids = []
for i, (num, t, s) in enumerate(et_clean):
    b1_ids.append(step(B1, 16, 46 + i * (SH + GAP), CW, SH, num, t, s, q))
for i in range(len(b1_ids) - 1):
    edge(b1_ids[i], b1_ids[i + 1], ARROW, parent=B1)

# ============================= Row 2: single convergence point =============================
a = PAL["aln"]
ALIGN = vertex(20, 608, 1270, 70,
               "<b>Temporal alignment</b> -- both streams trimmed to the shared overlap window using "
               "common stimulus-onset markers; each stream is then epoched independently at its OWN "
               "native rate (300 Hz EEG / 120 Hz eye-tracking) -- neither is resampled to the other's rate",
               "rounded=1;arcSize=8;fillColor=" + a["fill"] + ";strokeColor=" + a["head"] +
               ";strokeWidth=2;html=1;verticalAlign=middle;align=center;fontColor=" + a["head"] +
               ";fontStyle=1;whiteSpace=wrap;")
edge(a1_ids[-1], ALIGN, "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#2E5AAC;strokeWidth=2;"
                        "endArrow=block;endFill=1;exitX=0.25;exitY=1;entryX=0.25;entryY=0;")
edge(b1_ids[-1], ALIGN, "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#3C7A2E;strokeWidth=2;"
                        "endArrow=block;endFill=1;exitX=0.75;exitY=1;entryX=0.75;entryY=0;")

# ============================= Row 3: per-modality epoching / feature construction =============================
A2 = swimlane(20, 708, 610, 386, "EEG Epoching -> Graph + Band-Power Features", p)
eeg_post = [
    ("7", "Epoching", "5.0 s window from stimulus onset -> 1500 samples/epoch"),
    ("8", "Channel harmonization", "zero-fill any missing position back to 24-wide, then drop the 5 "
     "non-cortical channels (X1, X2, X3, A2, TRG) -> 19 canonical scalp channels"),
    ("9", "Per-subject z-score normalization", "fit without labels (subject-level, transductive)"),
]
a2_ids = []
for i, (num, t, s) in enumerate(eeg_post):
    a2_ids.append(step(A2, 16, 46 + i * (SH + GAP), CW, SH, num, t, s, p))
for i in range(len(a2_ids) - 1):
    edge(a2_ids[i], a2_ids[i + 1], ARROW, parent=A2)

bw = (CW - 14) / 2
a_graph = step(A2, 16, 280, bw, 90, "10", "Dynamic graph construction",
               "ten 0.5 s sub-windows/epoch; Pearson correlation r_ij(t) per window -> edges "
               "(threshold tau = 0.30)", p)
a_bandpower = step(A2, 16 + bw + 14, 280, bw, 90, "10'", "Band-power extraction",
                    "5-band relative power per 0.5 s sub-window -- independent of the graph, "
                    "same features feed both branches", p)
edge(a2_ids[-1], a_graph, ARROW, parent=A2)
edge(a2_ids[-1], a_bandpower, ARROW, parent=A2)

B2 = swimlane(680, 708, 610, 386, "Eye-Tracking Epoching -> Model Inputs", q)
b_epoch = step(B2, 16, 46, CW, SH, "5", "Epoching",
               "5.0 s window from stimulus onset -> 600 samples/epoch", q)
b_sel = step(B2, 16, 280, bw, 90, "6", "Channel selection",
             "truncate to 3 channels (gaze x, gaze y, pupil) for the Transformer attention encoder", q)
b_roi = step(B2, 16 + bw + 14, 280, bw, 90, "7", "ROI dwell vector",
             "5x2 spatial grid occupancy histogram -> 10-dim ROI vector; modulates EEG graph "
             "attention in the full model only", q)
edge(b_epoch, b_sel, ARROW, parent=B2)
edge(b_epoch, b_roi, ARROW, parent=B2)

edge(ALIGN, a2_ids[0], "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#2E5AAC;strokeWidth=2;"
                       "endArrow=block;endFill=1;exitX=0.25;exitY=1;entryX=0.5;entryY=0;")
edge(ALIGN, b_epoch, "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#3C7A2E;strokeWidth=2;"
                     "endArrow=block;endFill=1;exitX=0.75;exitY=1;entryX=0.5;entryY=0;")

# ============================= Row 4: feeds Fig. 2 =============================
o = PAL["out"]
OUTROW = vertex(20, 1124, 1270, 100,
                "<div style='text-align:center'><b style='color:#B23A48'>Feeds Fig. 2 -- INS-HDGS-CMT "
                "Architecture</b><br/><span style='color:#2E5AAC'>Graph -&gt; Panel A.2 (DynamicGAT)  "
                "&nbsp;|&nbsp;  Band-power features -&gt; Panel A.3&#8242; (LIF Spiking Encoder), "
                "independently of the graph</span><br/><span style='color:#3C7A2E'>3-ch gaze/pupil -&gt; "
                "Panel B.1 (Transformer)  &nbsp;|&nbsp;  ROI vector -&gt; dashed link into Panel A.2 "
                "(full model only)</span></div>",
                "rounded=1;arcSize=8;fillColor=" + o["fill"] + ";strokeColor=" + o["head"] +
                ";strokeWidth=2;html=1;verticalAlign=middle;whiteSpace=wrap;")
edge(a_graph, OUTROW, EEG_ARROW + "exitX=0.5;exitY=1;entryX=0.18;entryY=0;")
edge(a_bandpower, OUTROW, EEG_ARROW + "exitX=0.5;exitY=1;entryX=0.4;entryY=0;")
edge(b_sel, OUTROW, ET_ARROW + "exitX=0.5;exitY=1;entryX=0.6;entryY=0;")
edge(b_roi, OUTROW, ET_ARROW + "exitX=0.5;exitY=1;entryX=0.82;entryY=0;")

# ============================= Row 5: label computation (standalone, no forward-pass arrows) =============================
l = PAL["lbl"]
LBL = vertex(20, 1254, 1270, 90,
             "<div style='text-align:center'><b style='color:#5C6670'>Production label: "
             "engagement_phase3d (supervisory target ONLY -- not a model input, not part of the "
             "forward pass above)</b><br/><span style='color:#444'>Computed from the same epoched "
             "streams: mean(P3, C3, F3, Fz) -&gt; Welch band power (theta 4-8, alpha 8-13, beta 13-30 "
             "Hz) + theta/beta ratio + P3-Fz asymmetry, combined with epoched gaze-derived terms "
             "-&gt; HIGH / LOW label</span></div>",
             "rounded=1;arcSize=8;fillColor=" + l["fill"] + ";strokeColor=" + l["head"] +
             ";strokeWidth=2;html=1;verticalAlign=middle;dashed=1;whiteSpace=wrap;")

note = nid()
cells.append(f'<mxCell id="{note}" value="Note: common-average referencing (EEG step 3) and ICA (step 6) run '
             f'on each subject&#8217;s own post-bad-channel-removal set (24 channels for every subject except '
             f'S01); the 5 non-cortical channels are zero-filled then dropped only at harmonization '
             f'(step 8), after signal preprocessing -- not before. Channel selection and the ROI dwell '
             f'vector are independent, parallel outputs of eye-tracking epoching, not a sequential chain." '
             f'style="text;html=1;align=center;fontStyle=2;fontSize=11;fontColor=#444444;whiteSpace=wrap;" '
             f'vertex="1" parent="1"><mxGeometry x="20" y="1354" width="1270" height="40" as="geometry"/>'
             f'</mxCell>')

xml = f'''<mxfile host="app.diagrams.net">
  <diagram name="Preprocessing-Pipeline" id="{uuid.uuid4()}">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1"
        fold="1" page="1" pageScale="1" pageWidth="1310" pageHeight="1410" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {"".join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
OUT.write_text(xml, encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes,", len(cells), "cells")
