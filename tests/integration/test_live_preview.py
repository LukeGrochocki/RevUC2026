"""
Live / integration checks against real APIs.

Run (from repo root, with `.env` filled in):

  GARDN_INTEGRATION=1 .venv/bin/pytest tests/integration -v -s

Artifacts are written under tests/integration/output/ (gitignored) so you can open
`preview_*.html` in a browser and inspect `last_response.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _integration_enabled() -> bool:
    return os.getenv("GARDN_INTEGRATION", "").lower() in ("1", "true", "yes")


def _has_all_keys() -> bool:
    if not os.getenv("FEATHERLESS_API_KEY"):
        return False
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return False
    if not os.getenv("SCREENSHOTONE_ACCESS_KEY"):
        return False
    return True


@pytest.fixture
def integration_gate():
    if not _integration_enabled():
        pytest.skip("Set GARDN_INTEGRATION=1 to run live API tests (costs quota).")
    if not _has_all_keys():
        pytest.skip(
            "Need FEATHERLESS_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), "
            "SCREENSHOTONE_ACCESS_KEY in environment."
        )


@pytest.mark.integration
def test_live_pipeline_example_sites_writes_artifacts(integration_gate):
    """Calls ScreenshotOne → Gemini → Featherless; saves JSON + HTML files to tests/integration/output/."""
    from garden_preview import run_garden_preview_pipeline

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://example.com",
        "https://www.wikipedia.org/",
    ]
    user_notes = (
        "Focus on the main heading and primary navigation or links; "
        "describe distinct UI regions."
    )

    result = run_garden_preview_pipeline(
        urls=urls,
        user_notes=user_notes,
        screenshots_b64=[],
    )

    # Persist full response (may include large raw model JSON — fine for local inspection)
    json_path = OUTPUT_DIR / "last_response.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    previews = result.get("previews") or []
    for i, p in enumerate(previews):
        html_fragment = p.get("preview_html") or ""
        title = p.get("title", f"part_{i}")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:60]
        wrap = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Preview: {title}</title>
  <style>body {{ font-family: system-ui, sans-serif; margin: 1rem; }}</style>
</head>
<body>
  <h1>{title}</h1>
  <p><small>source: {p.get("source_url", "")}</small></p>
  <hr/>
  {html_fragment}
</body>
</html>
"""
        (OUTPUT_DIR / f"preview_{i}_{safe_name}.html").write_text(wrap, encoding="utf-8")

    # Console summary for pytest -s
    print("\n========== GARDN LIVE PREVIEW ==========")
    print("screenshot_source:", result.get("screenshot_source"))
    print("previews count:", len(previews))
    for i, p in enumerate(previews):
        print(f"  [{i}] {p.get('title')!r} — {p.get('source_url')!r}")
        vs = p.get("visual_summary")
        if vs:
            print(f"       visual: {vs[:120]}...")
    print(f"Wrote: {json_path}")
    print(f"Open HTML files in: {OUTPUT_DIR}")
    print("========================================\n")

    assert result.get("screenshot_source") == "screenshotone"
    assert len(previews) >= 1
    assert json_path.is_file()
