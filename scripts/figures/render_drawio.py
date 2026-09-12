#!/usr/bin/env python3
"""
Render any .drawio (mxGraph) file to PDF/PNG using headless Chrome/Edge and draw.io's own
open-source viewer script (https://viewer.diagrams.net/js/viewer-static.min.js). Needs Chrome or
Edge installed and one-time internet access to load that script (no server round-trip of the
diagram itself; it renders entirely client-side from the local .drawio file).

Generalized version of render_fig2_drawio.py -- takes an explicit input path so it can render any
figure in paper/figures/, not just fig2_Architecture.drawio.

Equivalent manual alternative, no scripting: open the .drawio at https://app.diagrams.net
(File > Open From > Device) and use File > Export as > PNG/PDF.

Run:  python scripts/figures/render_drawio.py paper/figures/fig_preprocessing.drawio
      python scripts/figures/render_drawio.py paper/figures/fig_preprocessing.drawio --browser "C:/Program Files/Microsoft/Edge/Application/msedge.exe"
"""
import argparse
import html
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageChops

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]


def find_browser(explicit):
    if explicit:
        return explicit
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("No Chrome/Edge found; pass --browser <path to chrome.exe or msedge.exe>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to .drawio file")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--scale", type=float, default=2.5, help="device scale factor (print resolution)")
    ap.add_argument("--window", default="1500,1500")
    ap.add_argument("--out-dir", default=None, help="defaults to the input file's directory")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else in_path.parent
    stem = in_path.stem

    browser = find_browser(args.browser)
    xml = in_path.read_text(encoding="utf-8")
    cfg = json.dumps({"xml": xml, "resize": True, "toolbar": None})
    attr = html.escape(cfg, quote=True)
    page = (f'<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#fff;">'
            f'<div class="mxgraph" style="max-width:100%;" data-mxgraph="{attr}"></div>'
            f'<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>'
            f'</body></html>')
    html_path = out_dir / f"_{stem}_viewer_tmp.html"
    html_path.write_text(page, encoding="utf-8")

    shot_path = out_dir / f"_{stem}_shot_tmp.png"
    subprocess.run([
        browser, "--headless", "--disable-gpu", "--no-sandbox",
        f"--window-size={args.window}", f"--force-device-scale-factor={args.scale}",
        "--virtual-time-budget=8000", f"--screenshot={shot_path}", f"file:///{html_path.as_posix()}",
    ], check=True, capture_output=True)

    im = Image.open(shot_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        raise SystemExit("Rendered page is blank; the viewer script may not have loaded (check internet access)")
    pad = 20
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(im.size[0], x1 + pad), min(im.size[1], y1 + pad)
    clipped = y1 >= im.size[1] - 2 or x1 >= im.size[0] - 2
    crop = im.crop((x0, y0, x1, y1))

    crop.save(out_dir / f"{stem}.png", dpi=(300, 300))
    crop.save(out_dir / f"{stem}.pdf", "PDF", resolution=300)
    html_path.unlink(missing_ok=True)
    shot_path.unlink(missing_ok=True)

    print(f"wrote {out_dir / (stem + '.png')} and .pdf, {crop.size}")
    if clipped:
        print("WARNING: content touched the render-window edge; re-run with a larger --window")


if __name__ == "__main__":
    main()
