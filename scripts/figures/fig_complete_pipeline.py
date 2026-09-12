#!/usr/bin/env python3
"""
Figure -- INS-HDGS-CMT complete pipeline (preprocessing + architecture) as a single .drawio
(mxGraph) source, laid out to match this manuscript's own established figure conventions (see
fig2_architecture.py / fig_preprocessing.py): a ~1270-unit-wide stack of swimlanes, not a single
wide row. A first version of this figure used one very wide row (3420 x 1090 units) mirroring a
reference block diagram; at this manuscript's fixed \\includegraphics[width=\\textwidth] embedding
(every figure, no exceptions -- see paper/*.tex), that aspect ratio would have printed at an
illegible ~2.5-3.5pt font. This version keeps the exact same technical content and both
corrections from that pass (see below) but wraps the architecture into panels B/C (side-by-side
branches, like fig2's DynamicGAT/LIF panel) instead of one row.

Corrections carried over from the code verification against src/model/models/*.py:
  1. The graph module is plain `DynamicGAT` (models/dynamic_gat.py) -- there is no "Hierarchical"
     variant in the code. Its per-window GAT and temporal self-attention (`self.temporal`, an
     nn.TransformerEncoder) are two stages of ONE module -- drawn here as one box.
  2. ROI attention does NOT gate the eye-tracking embedding z_ET -- z_ET comes straight out of the
     Transformer attention encoder. Two distinct ROI mechanisms act on the EEG side: (a) the
     Transformer attention encoder's own learned ROI-attention head, et_roi_attn = roi_proj(et_emb)
     -- NOT the precomputed ROI saliency vector r -- modulates the graph adjacency before DynamicGAT
     whenever the ET attention pathway is active, i.e. always in the full model
     (models/roi_modulation.py; ins_hdgs_cmt.py: `mod_roi = et_roi_attn if et_roi_attn is not None
     else roi_vector`); and (b) the precomputed ROI saliency vector r itself gates the merged EEG
     embedding after concat(z_Graph, spiking) (models/roi_attention.py).

Tensor shapes / hyperparameters (LIF: 2 layers x 128 units x 10 steps, the rule-key projection / 8
soft rules / rule evidence R / bypass logits / learned gate alpha formula, and the
validation-derived decision threshold) verified against models/ins_hdgs_cmt.py,
models/spiking_encoder.py, models/neuro_symbolic.py and evaluation/losocv.py.

Panel lettering (A-F) is local to this figure, per this manuscript's per-figure lettering
convention (each figure letters its own panels from A).

This is the manuscript's Fig. 2 (INS_HDGS_CMT_manuscript.tex, \label{fig2}), superseding the
older 5-panel fig2_architecture.py (which drew the now-corrected ROI-gating mistake in point 2
above and omitted preprocessing). Output: paper/figures/fig2_Architecture.drawio. Render with
scripts/figures/render_drawio.py.

Run:  python scripts/figures/fig_complete_pipeline.py
"""
import base64
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "paper" / "figures" / "fig2_assets"
OUT = ROOT / "paper" / "figures" / "fig2_Architecture.drawio"


def b64(name):
    return base64.b64encode((ASSETS / name).read_bytes()).decode("ascii")

PAL = {
    "eeg": dict(head="#2E5AAC", fill="#EAF0FB"),
    "et":  dict(head="#3C7A2E", fill="#EEF6EA"),
    "fus": dict(head="#6A4C93", fill="#F1ECF8"),
    "dec": dict(head="#C9822A", fill="#FCF1E2"),
    "out": dict(head="#B23A48", fill="#FBE7E9"),
    "aln": dict(head="#555555", fill="#EFEFEF"),
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


def edge(src, tgt, style="", parent="1", label="", waypoints=None):
    cid = nid()
    pts = ""
    if waypoints:
        pts_str = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in waypoints)
        pts = f'<Array as="points">{pts_str}</Array>'
    cells.append(f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" edge="1" parent="{parent}" '
                 f'source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
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


def text_card(parent, x, y, w, h, title, sub, pal, fill="#FFFFFF"):
    box = nid()
    sub_html = escape(sub).replace("\n", "&lt;br&gt;")
    cells.append(f'<mxCell id="{box}" value="&lt;b&gt;{escape(title)}&lt;/b&gt;&lt;br&gt;'
                 f'&lt;span style=&quot;font-size:10.5px;color:#5B6B7A;font-family:Consolas,Menlo,monospace&quot;'
                 f'&gt;{sub_html}&lt;/span&gt;" '
                 f'style="rounded=1;arcSize=10;fillColor={fill};strokeColor=#DADFE3;strokeWidth=1;align=left;'
                 f'verticalAlign=top;spacingLeft=10;spacingTop=8;fontColor={pal["head"]};fontSize=12.5;html=1;'
                 f'fontFamily=Helvetica;whiteSpace=wrap;shadow=1;" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return box


def image_card(parent, x, y, w, h, img_name, title, sub, pal, fill="#FFFFFF"):
    """Same as text_card but with a real per-stage thumbnail (S24, epoch 11) on the left."""
    box = nid()
    thumb = min(h - 12, 64)
    cells.append(f'<mxCell id="{box}" value="" style="rounded=1;arcSize=10;fillColor={fill};'
                 f'strokeColor=#DADFE3;strokeWidth=1;shadow=1;" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    img = nid()
    data = b64(img_name)
    cells.append(f'<mxCell id="{img}" value="" style="shape=image;imageAspect=1;rounded=1;arcSize=20;'
                 f'strokeColor=#DADFE3;image=data:image/png,{data};" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x + 8}" y="{y + (h - thumb) / 2}" width="{thumb}" height="{thumb}" '
                 f'as="geometry"/></mxCell>')
    sub_html = escape(sub).replace("\n", "&lt;br&gt;")
    txt = nid()
    cells.append(f'<mxCell id="{txt}" value="&lt;b&gt;{escape(title)}&lt;/b&gt;&lt;br&gt;'
                 f'&lt;span style=&quot;font-size:10.5px;color:#5B6B7A;font-family:Consolas,Menlo,monospace&quot;'
                 f'&gt;{sub_html}&lt;/span&gt;" style="text;html=1;align=left;verticalAlign=middle;'
                 f'fontColor={pal["head"]};fontSize=12.5;fontFamily=Helvetica;whiteSpace=wrap;spacingLeft=4;" '
                 f'vertex="1" parent="{parent}"><mxGeometry x="{x + thumb + 18}" y="{y}" '
                 f'width="{w - thumb - 26}" height="{h}" as="geometry"/></mxCell>')
    return box


def oval(parent, x, y, w, h, label, pal):
    cid = nid()
    cells.append(f'<mxCell id="{cid}" value="{escape(label)}" style="ellipse;fillColor={pal["fill"]};'
                 f'strokeColor={pal["head"]};strokeWidth=2;fontStyle=1;fontColor={pal["head"]};fontSize=13;'
                 f'fontFamily=Consolas,Menlo,monospace;html=1;whiteSpace=wrap;shadow=1;" vertex="1" '
                 f'parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
                 f'</mxCell>')
    return cid


ARROW = "edgeStyle=none;rounded=1;html=1;strokeColor=#3C4650;strokeWidth=2.25;endArrow=blockThin;endFill=1;endSize=8;"
XARROW = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#3C4650;strokeWidth=2.25;endArrow=blockThin;endFill=1;endSize=8;"
DASH = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#6A3FA0;strokeWidth=2.25;dashed=1;endArrow=blockThin;endFill=1;endSize=8;"

title_id = nid()
cells.append(f'<mxCell id="{title_id}" value="INS-HDGS-CMT: Preprocessing and Complete Architecture" '
             f'style="text;html=1;align=center;fontStyle=1;fontSize=22;" vertex="1" parent="1">'
             f'<mxGeometry x="20" y="10" width="1270" height="36" as="geometry"/></mxCell>')

# =====================================================================================
# A. Preprocessing, synchronization, epoching
# =====================================================================================
g = PAL["aln"]
A = swimlane(20, 60, 1270, 240, "A.  Signal Preprocessing, Synchronisation and Epoch Construction", g)
a_raw_eeg = text_card(A, 16, 46, 150, 60, "Raw EEG", "19 ch - 300 Hz", PAL["eeg"])
a_eeg_clean = text_card(A, 186, 36, 270, 80, "EEG cleaning", "", PAL["eeg"])
a_raw_et = text_card(A, 16, 140, 150, 60, "Raw ET", "3 ch - 120 Hz", PAL["et"])
a_et_clean = text_card(A, 186, 130, 270, 80, "ET cleaning", "", PAL["et"])
a_align = text_card(A, 476, 46, 250, 164, "Temporal alignment + fixed epoching", "5.0 s epochs, native rates",
                     g, fill=g["fill"])
a_clean_eeg = text_card(A, 746, 46, 170, 60, "Clean EEG epoch", "1500 x 19", PAL["eeg"])
a_clean_et = text_card(A, 746, 140, 170, 60, "Clean ET epoch", "600 x 3", PAL["et"])
a_paired = text_card(A, 936, 76, 318, 104, "Paired clean 5-s tensors", "", g, fill=g["fill"])

for s, t in [(a_raw_eeg, a_eeg_clean), (a_raw_et, a_et_clean), (a_eeg_clean, a_align), (a_et_clean, a_align),
             (a_align, a_clean_eeg), (a_align, a_clean_et), (a_clean_eeg, a_paired), (a_clean_et, a_paired)]:
    edge(s, t, ARROW, parent=A)

# =====================================================================================
# B. EEG Graph-Spiking Pathway
# =====================================================================================
p = PAL["eeg"]
B = swimlane(20, 320, 610, 635, "B.  EEG Graph-Spiking Pathway", p)
bw = (578 - 14) / 2
b1 = text_card(B, 16, 46, 578, 50, "Clean EEG epoch", "1500 x 19 (from panel A)", p)
b2 = text_card(B, 16, 108, 578, 50, "10 x 0.5-s windows", "150 samples / window", p)
b3a = image_card(B, 16, 170, bw, 70, "a1_filtered_eeg.png", "Node features", "19 x 5", p)
b3b = image_card(B, 16 + bw + 14, 170, bw, 70, "a2_electrode_connectivity.png",
                  "Dynamic graph construction", "tau = 0.30", p)
b4a = text_card(B, 16, 252, bw, 70, "Learned band-weighted sum", "", p)
b4b = image_card(B, 16 + bw + 14, 252, bw, 70, "a3_dynamic_brain_graph.png",
                  "DynamicGAT", "(incl. temporal self-attn.)", p)
b5a = image_card(B, 16, 334, bw, 70, "a4_lif_spike_raster.png", "2-layer LIF encoder", "128 x 10 steps", p)
b5b = text_card(B, 16 + bw + 14, 334, bw, 70, "Graph embedding", "z_Graph", p)
b6a = text_card(B, 16, 416, bw, 60, "Spiking embedding", "", p)
b6b = image_card(B, 16 + bw + 14, 416, bw, 60, "a5_eeg_contrastive_embedding.png",
                  "EEG projection", "concat(z_Graph, spiking)", p)
b7 = text_card(B, 16, 488, 578, 60, "ROI-attention gate", "(EEG side only)", p)
b8 = oval(B, 189, 560, 200, 55, "z_EEG", p)

for s, t in [(b1, b2)]:
    edge(s, t, ARROW, parent=B)
edge(b2, b3a, ARROW, parent=B, label="features branch")
edge(b2, b3b, ARROW, parent=B, label="independent branch")
edge(b3a, b4a, ARROW, parent=B)
edge(b3b, b4b, ARROW, parent=B)
edge(b4a, b5a, ARROW, parent=B)
edge(b4b, b5b, ARROW, parent=B)
edge(b5a, b6a, ARROW, parent=B)
edge(b5b, b6b, ARROW, parent=B)
edge(b6a, b6b, ARROW, parent=B)
edge(b6b, b7, ARROW, parent=B)
edge(b7, b8, ARROW, parent=B)

# =====================================================================================
# C. Eye-Tracking Attention Pathway
# =====================================================================================
q = PAL["et"]
C = swimlane(680, 320, 610, 635, "C.  Eye-Tracking Attention Pathway", q)
c1 = text_card(C, 16, 46, 578, 50, "Clean ET epoch", "600 x 3", q)
c4 = image_card(C, 16, 108, 578, 60, "b3_roi_attention.png", "ROI saliency vector", "r in R^10", q)
c2 = image_card(C, 16, 180, 578, 50, "b1_gaze_trajectory.png", "ET sequence encoder input", "", q)
c3 = image_card(C, 16, 242, 578, 60, "b2_transformer_attention.png", "Transformer attention encoder", "", q)
c5 = oval(C, 189, 560, 200, 55, "z_ET", q)

edge(c1, c4, ARROW, parent=C, label="ROI branch")
edge(c1, c2, ARROW, parent=C)
edge(c2, c3, ARROW, parent=C)
edge(c3, c5, ARROW, parent=C, label="not ROI-gated")

# explicit waypoints route both dashed edges through the empty gutter between panels B and C,
# instead of mxGraph's default routing which cut across panel C's own boxes
edge(c3, b4b, DASH + "exitX=0;exitY=0.5;entryX=1;entryY=0.5;labelBackgroundColor=#FFFFFF;fontSize=10;",
     label="ROI graph modulation (full model only)", waypoints=[(650, 592), (650, 607)])
edge(c4, b7, DASH + "exitX=0;exitY=0.5;entryX=1;entryY=0.5;",
     label="ROI gate on merged EEG embedding", waypoints=[(660, 458), (660, 838)])

# =====================================================================================
# D. NeuroFusion Transformer
# =====================================================================================
f = PAL["fus"]
D = swimlane(20, 975, 1270, 150, "D.  NeuroFusion Transformer", f)
d1 = text_card(D, 16, 46, 820, 90, "3-token self-attention + directed cross-attention",
               "Q = EEG, K,V = Graph / ET", f)
d2 = oval(D, 870, 61, 200, 60, "z_Fusion", f)
edge(d1, d2, ARROW, parent=D)
# These three incoming edges target d1 (the actual "3-token self-attention..." box) directly, not
# the D swimlane container -- targeting the swimlane makes the arrowhead land on the panel's own
# header bar with no visible connection to any block inside it.
edge(b8, d1, XARROW + "exitX=0.5;exitY=1;entryX=0.2;entryY=0;")
edge(c5, d1, XARROW + "exitX=0.5;exitY=1;entryX=0.85;entryY=0;")
# z_Graph (raw DynamicGAT output, b5b) is passed into NeuroFusionTransformer.forward() as its own
# distinct 3rd positional argument (ins_hdgs_cmt.py: `g_in = graph_emb ...; self.fusion(eeg_emb,
# g_in, et_emb, ...)`), separately from z_EEG (which already absorbed graph_emb via eeg_merge) --
# routed through the B/C gutter, offset from the two ROI dashed edges that also use that gutter.
edge(b5b, d1, XARROW + "exitX=1;exitY=0.5;entryX=0.5;entryY=0;",
     label="z_Graph (direct)", waypoints=[(640, 689), (640, 965)])

# =====================================================================================
# E. Neuro-Symbolic Decision Layer
# =====================================================================================
d = PAL["dec"]
E = swimlane(20, 1145, 1270, 230, "E.  Neuro-Symbolic Decision Layer", d)
ew = (1238 - 28) / 3
e1 = text_card(E, 16, 46, ew, 70, "Rule-key projection", "", d)
e2 = text_card(E, 16 + ew + 14, 46, ew, 70, "8 soft rules", "", d)
e3 = text_card(E, 16 + 2 * (ew + 14), 46, ew, 70, "Rule evidence R", "", d)
e4 = text_card(E, 16, 128, ew, 80, "Bypass logits", "", d)
e5 = text_card(E, 16 + ew + 14, 128, ew * 2 + 14, 80, "Learned rule-bypass gate",
               "alpha . bypass + (1 - alpha) . R", d)
edge(e1, e2, ARROW, parent=E)
edge(e2, e3, ARROW, parent=E)
edge(e3, e5, ARROW, parent=E)
edge(e4, e5, ARROW, parent=E)
edge(d2, e1, XARROW + "exitX=0.3;exitY=1;entryX=0.5;entryY=0;")
# d2->e4 previously shared d2->e1's exit/entry coordinates, so orthogonal auto-routing rendered
# them collinear (one line appearing to pass through Rule-key projection on its way to Bypass
# logits, with no visible fork -- confirmed on the actual rendered PNG). Bypass logits sits directly
# below Rule-key projection, so any straight-down path would cut through that box; routed around
# panel E's own left inset instead (hugging the swimlane edge, not the outer page margin).
edge(d2, e4, XARROW + "exitX=0.9;exitY=1;entryX=0;entryY=0.5;",
     waypoints=[(1070, 1135), (25, 1135), (25, 1313)])

# =====================================================================================
# F. Output
# =====================================================================================
o = PAL["out"]
f1 = text_card(1, 195, 1395, 280, 70, "Softmax probabilities", "P(LOW), P(HIGH)", o, fill=o["fill"])
f2 = text_card(1, 555, 1395, 280, 70, "Decision threshold", "validation-derived", o, fill=o["fill"])
f3 = vertex(340, 1495, 630, 110,
            "<div style='text-align:center'><b style='color:#B23A48;font-size:15px'>F.  Engagement "
            "Prediction</b><br/><span style='color:#2E5AAC;font-weight:bold;font-size:16px'>LOW</span> "
            "<span style='color:#444'>vs</span> "
            "<span style='color:#B23A48;font-weight:bold;font-size:16px'>HIGH</span></div>",
            "rounded=1;arcSize=8;fillColor=" + o["fill"] + ";strokeColor=" + o["head"] +
            ";strokeWidth=2;html=1;verticalAlign=middle;")
edge(e5, f1, XARROW + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
edge(f1, f2, XARROW)
edge(f2, f3, XARROW + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram name="Complete-Pipeline" id="{uuid.uuid4()}">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1"
        fold="1" page="1" pageScale="1" pageWidth="1310" pageHeight="1630" math="0" shadow="0">
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
