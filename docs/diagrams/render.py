"""Render one diagram HTML to PNG. Usage: python3 docs/diagrams/render.py <src.html> <out.png>
Uses Playwright only. No API calls."""
import pathlib, sys
from playwright.sync_api import TimeoutError as PWT, sync_playwright

src, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(device_scale_factor=2, viewport={"width": 1280, "height": 900})
    page.goto(f"file://{pathlib.Path(src).resolve()}", wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PWT:
        page.evaluate("window.stop()"); page.wait_for_timeout(4000)
        print("warning: webfonts stalled, fallback typography", file=sys.stderr)
    page.evaluate("document.fonts.ready")
    svg = page.locator("svg").first
    svg.evaluate("el => { for (let a = el.parentElement; a; a = a.parentElement) a.style.setProperty('overflow', 'visible', 'important'); }")
    svg.screenshot(path=out, omit_background=True)
    b.close()
print("wrote", out)
