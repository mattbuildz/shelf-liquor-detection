"""Build the dark variant and both PNGs for every diagram.

Usage, from the repo root:  python3 docs/diagrams/build.py [name ...]

Reads docs/diagrams/<name>.html (the light source), writes <name>-dark.html next to it
by swapping the colour tokens, then renders assets/<name>-light.png and assets/<name>-dark.png.
Uses Playwright only. It touches no data, no model and no API. Playwright is not a project
dependency, so run this with a Python that has it (not `uv run`).
"""
import pathlib
import sys

from playwright.sync_api import TimeoutError as PWT, sync_playwright

ROOT = pathlib.Path("docs/diagrams")
OUT = pathlib.Path("assets")
NAMES = ["one-change", "problems", "error-anatomy", "pipeline", "experiment-loop", "score-coverage"]

# light token line -> dark token line. Every diagram declares the same :root block.
SWAP = {
    "--color-paper:   #f5f5f5": "--color-paper:   #2d3142",
    "--color-paper-2: #ececec": "--color-paper-2: #393e53",
    "--color-ink:     #2d3142": "--color-ink:     #f5f5f5",
    "--color-muted:   #4f5d75": "--color-muted:   #bfc0c0",
    "--color-soft:    #7a8399": "--color-soft:    #8e98ac",
    "--color-rule:    rgba(45,49,66,0.12)": "--color-rule:    rgba(245,245,245,0.16)",
    "--color-accent:  #eb6c36": "--color-accent:  #f08a59",
    "--color-accent-tint: rgba(235,108,54,0.08)": "--color-accent-tint: rgba(240,138,89,0.12)",
    "--color-card:    #ffffff": "--color-card:    #393e53",
    "--color-card-stroke: rgba(45,49,66,0.22)": "--color-card-stroke: rgba(245,245,245,0.22)",
}

# tokens only some diagrams declare
OPTIONAL = {
    "--color-input:   #ececec": "--color-input:   #2d3142",
    "--color-input-stroke: rgba(45,49,66,0.34)": "--color-input-stroke: rgba(245,245,245,0.38)",
    "--color-panel:   #2d3142": "--color-panel:   #22263a",
}


def write_dark(name):
    dark = (ROOT / f"{name}.html").read_text()
    swap = dict(SWAP)
    swap[f"{name}-title"] = f"{name}-dark-title"
    swap[f"{name}-desc"] = f"{name}-dark-desc"
    for light, replacement in swap.items():
        assert light in dark, f"{name}: missing {light}"
        dark = dark.replace(light, replacement)
    for light, replacement in OPTIONAL.items():
        dark = dark.replace(light, replacement)
    (ROOT / f"{name}-dark.html").write_text(dark)


def render(browser, src, out):
    page = browser.new_page(device_scale_factor=2, viewport={"width": 1280, "height": 900})
    page.goto(f"file://{src.resolve()}", wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PWT:
        page.evaluate("window.stop()")
        page.wait_for_timeout(4000)
        print("warning: webfonts stalled, fallback typography", file=sys.stderr)
    page.evaluate("document.fonts.ready")
    loaded = page.evaluate(
        "[...document.fonts].filter(f => f.status == 'loaded').map(f => f.family)"
        ".filter((v, i, a) => a.indexOf(v) == i).join(', ')"
    )
    svg = page.locator("svg").first
    svg.evaluate(
        "el => { for (let a = el.parentElement; a; a = a.parentElement)"
        " a.style.setProperty('overflow', 'visible', 'important'); }"
    )
    svg.screenshot(path=str(out), omit_background=True)
    page.close()
    print(f"wrote {out}  fonts: {loaded}")


names = sys.argv[1:] or NAMES
OUT.mkdir(exist_ok=True)
for name in names:
    write_dark(name)
with sync_playwright() as p:
    browser = p.chromium.launch()
    for name in names:
        render(browser, ROOT / f"{name}.html", OUT / f"{name}-light.png")
        render(browser, ROOT / f"{name}-dark.html", OUT / f"{name}-dark.png")
    browser.close()
