"""Put a recording into docs/visualizer/index.html and render the README GIFs.

  python tools/build_visualizer.py inference.json --fast-text "answer printed by the litemoe command"
  python tools/build_visualizer.py --gif          # needs playwright, Chromium and ffmpeg
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "docs" / "visualizer" / "index.html"
# The GIF keeps the parts that change each step, at a width a README shows without scaling.
GIF_CSS = """header, #controls .bar, #explain, #tokHint, #compare, footer, #kvnote, #routeNote, .tok.fold { display: none !important; }
body { padding: 10px; } main { gap: 8px; } .panel { padding: 10px; gap: 6px; } .stack { gap: 8px; } .grid3 { gap: 8px; }"""


def embed(recording, fast_text):
    data = json.loads(Path(recording).read_text(encoding="utf-8"))
    data["fast_text"] = fast_text
    line = "const D = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";"
    page, count = re.subn(r"^const D = .*;$", lambda _: line, PAGE.read_text(encoding="utf-8"), flags=re.M)
    assert count == 1, "recording line not found in the page"
    PAGE.write_text(page, encoding="utf-8")


def render_gif(lang, chromium):
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium)
        page = browser.new_page(viewport=dict(width=800, height=100), color_scheme="light")
        page.goto(f"{PAGE.as_uri()}?lang={lang}")
        page.add_style_tag(content=GIF_CSS)
        steps = page.evaluate("O")
        page.set_viewport_size(dict(width=800, height=page.evaluate("document.documentElement.scrollHeight")))
        # Size for the tallest frame so the picture doesn't jump.
        page.evaluate(f"go({steps - 1})")
        page.set_viewport_size(dict(width=800, height=page.evaluate("document.documentElement.scrollHeight")))
        for i in range(steps):
            page.evaluate(f"go({i})")
            page.screenshot(path=f"{tmp}/{i:03d}.png")
        browser.close()
        last = steps - 1
        for extra in range(1, 5):  # Hold the final answer before the loop restarts.
            Path(f"{tmp}/{last + extra:03d}.png").write_bytes(Path(f"{tmp}/{last:03d}.png").read_bytes())
        graph = "split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1.6", "-i", f"{tmp}/%03d.png", "-vf", graph,
                        str(PAGE.parent / f"demo.{lang}.gif")], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", nargs="?", help="output of tools/record_inference.py")
    p.add_argument("--fast-text", help="answer from the default fast path, shown next to the recorded answer")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--chromium", help="Chromium executable, if playwright's own isn't installed")
    args = p.parse_args()
    if args.recording:
        embed(args.recording, args.fast_text)
    if args.gif:
        for lang in ("zh", "en"):
            render_gif(lang, args.chromium)


if __name__ == "__main__":
    main()
