#!/usr/bin/env python3
"""
Run the full garden preview pipeline against real APIs and write artifacts under output/.

Requires .env: FEATHERLESS_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), SCREENSHOTONE_ACCESS_KEY

Usage (repo root):

  .venv/bin/python scripts/demo_preview.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"google\.genai\.types",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> None:
    from garden_preview import run_garden_preview_pipeline

    out = ROOT / "output"
    out.mkdir(exist_ok=True)

    urls = [
        "https://example.com",
        "https://example.org",
    ]
    user_notes = (
        "Call out the main heading and any obvious nav or links; "
        "we are sampling UI to remix into a garden preview."
    )

    print("Calling ScreenshotOne → Gemini → Featherless (this may take a minute)...")
    result = run_garden_preview_pipeline(
        urls=urls,
        user_notes=user_notes,
        screenshots_b64=[],
    )

    json_path = out / "demo_last_response.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    previews = result.get("previews") or []
    for i, p in enumerate(previews):
        html_fragment = p.get("preview_html") or ""
        title = p.get("title", f"part_{i}")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:50]
        page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>{title}</title>
<style>body{{font-family:system-ui;margin:1rem}}</style></head>
<body><h1>{title}</h1><p><small>{p.get("source_url","")}</small></p><hr/>{html_fragment}</body></html>
"""
        (out / f"demo_preview_{i}_{safe}.html").write_text(page, encoding="utf-8")

    print()
    print("screenshot_source:", result.get("screenshot_source"))
    print("previews:", len(previews))
    for i, p in enumerate(previews):
        print(f"  [{i}] {p.get('title')!r}")
        if p.get("visual_summary"):
            print(f"       visual_summary: {str(p['visual_summary'])[:100]}...")
    print()
    print(f"Wrote: {json_path}")
    print(f"HTML previews: {out}/demo_preview_*.html")
    print()


if __name__ == "__main__":
    main()
