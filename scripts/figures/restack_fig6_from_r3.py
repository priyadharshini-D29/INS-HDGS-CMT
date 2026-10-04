"""
Restack manuscript Figure 7 vertically from the published fig6_combined_r3.pdf itself.

fig6_combined_r3.pdf places panel A (critical-difference diagram) and panel B
(ROC + PR) side by side on one 23-in-wide page, which renders its smallest labels
at about 2.5-3.3 pt once scaled to \textwidth. This script clips the two panel
regions out of that very page (split at the midpoint between the 'A' and 'B'
labels) and places them one above the other, so the embedded vector content is
untouched and the figure is identical to r3 by construction, at roughly twice
the printed type size.

Run:  python scripts/figures/restack_fig6_from_r3.py <path-to-fig6_combined_r3.pdf> <out.pdf>
"""
import sys
import pymupdf

src_path, out_path = sys.argv[1], sys.argv[2]
src = pymupdf.open(src_path)
pg = src[0]
W, H = pg.rect.width, pg.rect.height

labels = {}
for x0, y0, x1, y1, word, *_ in pg.get_text("words"):
    if word in ("A", "B") and y1 < 30:
        labels[word] = (x0, y0, x1, y1)
if not ("A" in labels and "B" in labels):
    raise SystemExit(f"could not locate panel labels near the top: {labels}")

split = labels["B"][0] - 17        # mid-gap between the panels
clipA = pymupdf.Rect(0, 0, split, H)
clipB = pymupdf.Rect(split, 0, W, H)
wA, wB = split, W - split

gap, margin = 20, 6
W2 = max(wA, wB) + 2 * margin
H2 = 2 * H + gap + 2 * margin

out = pymupdf.open()
npg = out.new_page(width=W2, height=H2)
npg.show_pdf_page(pymupdf.Rect(margin, margin, margin + wA, margin + H), src, 0, clip=clipA)
npg.show_pdf_page(pymupdf.Rect(margin, margin + H + gap, margin + wB, margin + H + gap + H), src, 0, clip=clipB)
out.save(out_path)
print("wrote %s | %.0f x %.0f pt (panels %.0f / %.0f wide)" % (out_path, W2, H2, wA, wB))
