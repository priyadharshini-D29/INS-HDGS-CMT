"""
Assemble manuscript Figure 6 from its two committed panels:
  panel A = fig4_cd_eeg_headline.pdf  (Nemenyi critical-difference, EEG encoders)
  panel B = fig6_roc_pr_eeg.pdf       (leakage-free ROC + PR, top-3 EEG-only)
into a single wide fig6_combined.pdf (vector, panels placed side by side with
bold A / B labels).  Run after make_fig4_fig6.py:  python figures/combine_fig6.py
"""
import pymupdf
from pathlib import Path

FIGD = Path(__file__).resolve().parents[1] / "manuscript" / "figures"
A = pymupdf.open(FIGD / "fig4_cd_eeg_headline.pdf")
B = pymupdf.open(FIGD / "fig6_roc_pr_eeg.pdf")
ar, br = A[0].rect, B[0].rect

gap, lab, margin_r = 35, 24, 20
W = 15 + ar.width + gap + br.width + margin_r
H = lab + max(ar.height, br.height) + 12

out = pymupdf.open()
pg = out.new_page(width=W, height=H)

ax0 = 15
ay0 = lab + (H - lab - ar.height) / 2
pg.show_pdf_page(pymupdf.Rect(ax0, ay0, ax0 + ar.width, ay0 + ar.height), A, 0)

bx0 = ax0 + ar.width + gap
by0 = lab + (H - lab - br.height) / 2
pg.show_pdf_page(pymupdf.Rect(bx0, by0, bx0 + br.width, by0 + br.height), B, 0)

pg.insert_text((ax0, 17), "A", fontsize=15, fontname="hebo")
pg.insert_text((bx0, 17), "B", fontsize=15, fontname="hebo")

out.save(FIGD / "fig6_combined.pdf")
print("wrote %s | %.0f x %.0f pt" % (FIGD / "fig6_combined.pdf", W, H))
