"""Gardn API, pipeline helpers, and optional live integration (``GARDN_INTEGRATION=1``)."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import httpx
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from garden_preview import (
    _featherless_message_content,
    decode_base64_image,
    featherless_chat,
    featherless_preview_messages,
    gemini_vision_instruction_text,
    run_garden_preview_pipeline,
)
from main import _cors_allow_origins, app

load_dotenv()

# --- shared fixtures / dirs -------------------------------------------------

# Minimal valid 1×1 PNG (transparent pixel)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_GEMINI_MIN = {
    "screenshots": [
        {
            "url_index": 0,
            "url": "https://example.com",
            "criteria": "",
            "effective_analysis_target": "whole page",
            "summary": "mock visual",
            "preview_html": '<div style="padding:2px">gemini</div>',
            "preview_image_png_base64": None,
        }
    ]
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GARDN_DISABLE_HTML_PREVIEW", "1")
    return TestClient(app)


def _integration_enabled() -> bool:
    return os.getenv("GARDN_INTEGRATION", "").lower() in ("1", "true", "yes")


def _has_all_live_keys() -> bool:
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
    if not _has_all_live_keys():
        pytest.skip(
            "Need FEATHERLESS_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), "
            "SCREENSHOTONE_ACCESS_KEY in environment."
        )


def _safe_filename_segment(url: str, index: int) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "url").replace(":", "_")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in host)[:48]
    return f"{index:02d}_{safe}.png"


def _write_pngs_from_b64(
    out_dir: Path,
    urls: list[str],
    images_b64: list[str],
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    relative: list[str] = []
    for i, (url, b64) in enumerate(zip(urls, images_b64, strict=True)):
        name = _safe_filename_segment(url, i)
        path = out_dir / name
        path.write_bytes(base64.b64decode(b64))
        relative.append(str(path.relative_to(OUTPUT_DIR)))
    return relative


def _assert_png_file(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- unit: API + small helpers ----------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_cors_allow_origins_strips_spaces(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, http://example.com")
    assert _cors_allow_origins() == ["http://localhost:3000", "http://example.com"]


def test_featherless_message_content_string():
    data = {"choices": [{"message": {"content": "hello"}}]}
    assert _featherless_message_content(data) == "hello"


def test_featherless_message_content_missing_choices():
    with pytest.raises(RuntimeError, match="choices"):
        _featherless_message_content({})


@patch("garden_preview.httpx.stream")
def test_featherless_chat_retries_on_transport_error(mock_stream, monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    monkeypatch.setenv("FEATHERLESS_RETRIES", "2")
    bad = MagicMock()
    bad.__enter__.return_value = bad
    bad.__exit__.return_value = None
    bad.is_success = True
    bad.iter_lines.side_effect = httpx.RemoteProtocolError("connection dropped")

    good = MagicMock()
    good.__enter__.return_value = good
    good.__exit__.return_value = None
    good.is_success = True
    sse = "data: " + json.dumps(
        {"choices": [{"index": 0, "delta": {"content": "ok"}}]}
    )
    good.iter_lines.return_value = [sse, "data: [DONE]"]

    mock_stream.side_effect = [bad, good]

    out = featherless_chat(messages=[{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert mock_stream.call_count == 2
    assert mock_stream.call_args.kwargs["json"].get("stream") is True


@patch("garden_preview.httpx.stream")
@patch("garden_preview.httpx.post")
def test_featherless_chat_non_streaming_when_disabled(mock_post, mock_stream, monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    monkeypatch.setenv("FEATHERLESS_STREAM", "0")
    ok = MagicMock()
    ok.is_success = True
    ok.json.return_value = {"choices": [{"message": {"content": "sync"}}]}
    mock_post.return_value = ok

    out = featherless_chat(messages=[{"role": "user", "content": "hi"}])
    assert out == "sync"
    mock_post.assert_called_once()
    mock_stream.assert_not_called()


def test_decode_base64_image_data_url():
    raw = decode_base64_image(f"data:image/png;base64,{_TINY_PNG_B64}")
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@patch("google.genai.Client")
def test_gemini_analyze_uses_google_genai_sdk(mock_client_cls, monkeypatch):
    """Vision path uses `google-genai` (Client + Part.from_bytes)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GARDN_DISABLE_HTML_PREVIEW", "1")
    mock_resp = MagicMock()
    mock_resp.candidates = [MagicMock()]
    mock_resp.text = json.dumps(
        {
            "screenshots": [
                {
                    "url_index": 0,
                    "url": "https://example.com",
                    "criteria": "note",
                    "effective_analysis_target": "note",
                    "summary": "x",
                    "preview_html": "<div/>",
                    "preview_image_png_base64": None,
                }
            ]
        }
    )
    mock_inst = MagicMock()
    mock_inst.models.generate_content.return_value = mock_resp
    mock_client_cls.return_value = mock_inst

    from garden_preview import gemini_analyze_screenshots

    out = gemini_analyze_screenshots(
        sites=[{"url": "https://example.com", "criteria": "note"}],
        screenshots_b64=[_TINY_PNG_B64],
    )
    mock_client_cls.assert_called_once_with(api_key="test-key")
    mock_inst.models.generate_content.assert_called_once()
    call_kw = mock_inst.models.generate_content.call_args.kwargs
    assert call_kw["model"]
    contents = call_kw["contents"]
    assert isinstance(contents, list)
    assert len(contents) >= 3
    assert out["screenshots"][0]["url_index"] == 0


@patch("garden_preview.featherless_chat")
@patch("garden_preview.gemini_analyze_screenshots")
@patch("garden_preview.capture_screenshots_for_urls")
def test_garden_preview_legacy_urls_user_notes(
    mock_capture, mock_gemini, mock_chat, client
):
    """Legacy request shape: urls + user_notes applies the same criteria to each URL."""
    mock_capture.return_value = [_TINY_PNG_B64]
    mock_gemini.return_value = _GEMINI_MIN
    mock_chat.return_value = json.dumps(
        {
            "previews": [
                {
                    "title": "Navigation",
                    "source_url": "https://example.com",
                    "criteria": "nav",
                    "preview_html": "<div>n</div>",
                    "preview_image_png_base64": None,
                    "notes": "x",
                }
            ]
        }
    )
    r = client.post(
        "/api/garden-preview",
        json={"urls": ["https://example.com"], "user_notes": "nav", "screenshots": []},
    )
    assert r.status_code == 200
    mock_gemini.assert_called_once()
    call_kw = mock_gemini.call_args.kwargs
    assert call_kw["sites"] == [{"url": "https://example.com", "criteria": "nav"}]


@patch("garden_preview.featherless_chat")
@patch("garden_preview.gemini_analyze_screenshots")
@patch("garden_preview.capture_screenshots_for_urls")
def test_garden_preview_uses_screenshotone(
    mock_capture, mock_gemini, mock_chat, client
):
    mock_capture.return_value = [_TINY_PNG_B64]
    mock_gemini.return_value = _GEMINI_MIN
    mock_chat.return_value = json.dumps(
        {
            "previews": [
                {
                    "title": "Navigation",
                    "source_url": "https://example.com",
                    "criteria": "the nav bar",
                    "preview_html": '<div style="padding:8px">nav</div>',
                    "preview_image_png_base64": None,
                    "notes": "mock preview",
                }
            ]
        }
    )

    r = client.post(
        "/api/garden-preview",
        json={
            "sites": [{"url": "https://example.com", "criteria": "the nav bar"}],
            "screenshots": [],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["screenshot_source"] == "screenshotone"
    assert "previews" in body
    assert len(body["previews"]) == 1
    assert body["previews"][0]["title"] == "Navigation"
    assert body["previews"][0]["gemini"]["summary"] == "mock visual"
    mock_capture.assert_called_once_with(["https://example.com"])
    mock_gemini.assert_called_once()
    mock_chat.assert_called_once()


@patch("garden_preview.featherless_chat")
@patch("garden_preview.gemini_analyze_screenshots")
@patch("garden_preview.capture_screenshots_for_urls")
def test_garden_preview_client_screenshots_skip_screenshotone(
    mock_capture, mock_gemini, mock_chat, client
):
    mock_gemini.return_value = _GEMINI_MIN
    mock_chat.return_value = json.dumps(
        {
            "previews": [
                {
                    "title": "Navigation",
                    "source_url": "https://example.com",
                    "criteria": "nav",
                    "preview_html": "<div>nav</div>",
                    "preview_image_png_base64": None,
                    "notes": "mock",
                }
            ]
        }
    )

    r = client.post(
        "/api/garden-preview",
        json={
            "sites": [{"url": "https://example.com", "criteria": "nav"}],
            "screenshots": [_TINY_PNG_B64],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["screenshot_source"] == "client"
    assert body["previews"][0]["gemini"]["summary"] == "mock visual"
    mock_capture.assert_not_called()
    mock_gemini.assert_called_once()
    mock_chat.assert_called_once()


@patch("garden_preview.featherless_chat")
@patch("garden_preview.gemini_analyze_screenshots")
@patch("garden_preview.capture_screenshots_for_urls")
def test_garden_preview_include_model_debug(
    mock_capture, mock_gemini, mock_chat, client
):
    mock_capture.return_value = [_TINY_PNG_B64]
    mock_gemini.return_value = _GEMINI_MIN
    mock_chat.return_value = json.dumps(
        {
            "previews": [
                {
                    "title": "X",
                    "source_url": "https://a.test",
                    "criteria": "",
                    "preview_html": "<div/>",
                    "preview_image_png_base64": None,
                    "notes": "n",
                }
            ]
        }
    )
    r = client.post(
        "/api/garden-preview",
        json={
            "sites": [{"url": "https://a.test", "criteria": ""}],
            "screenshots": [],
            "include_model_debug": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("featherless_raw") is not None
    assert data.get("gemini_raw") is not None


@patch("garden_preview.featherless_chat")
@patch("garden_preview.gemini_analyze_screenshots")
@patch("garden_preview.capture_screenshots_for_urls")
def test_garden_preview_featherless_failure_returns_502(
    mock_capture, mock_gemini, mock_chat, client
):
    mock_capture.return_value = [_TINY_PNG_B64]
    mock_gemini.return_value = _GEMINI_MIN
    mock_chat.side_effect = RuntimeError("Featherless HTTP 503: unavailable")

    r = client.post(
        "/api/garden-preview",
        json={"sites": [{"url": "https://example.com", "criteria": ""}], "screenshots": []},
    )
    assert r.status_code == 502
    assert "unavailable" in r.json()["detail"]


@patch("garden_preview.requests.get")
def test_capture_screenshots_for_urls_success(mock_get, monkeypatch):
    monkeypatch.setenv("SCREENSHOTONE_ACCESS_KEY", "test-access-key")
    from garden_preview import capture_screenshots_for_urls

    png_bytes = __import__("base64").b64decode(_TINY_PNG_B64)
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.headers = {"Content-Type": "image/png"}
    mock_resp.content = png_bytes
    mock_get.return_value = mock_resp

    out = capture_screenshots_for_urls(["https://example.com"])
    assert len(out) == 1
    assert out[0] == _TINY_PNG_B64
    mock_get.assert_called_once()
    call_kw = mock_get.call_args
    assert "api.screenshotone.com/take" in call_kw[0][0]
    params = call_kw[1]["params"]
    assert params["url"] == "https://example.com"
    assert params["access_key"] == "test-access-key"


@patch("garden_preview.time.sleep")
@patch("garden_preview.requests.get")
def test_capture_screenshots_retries_on_500(mock_get, _mock_sleep, monkeypatch):
    monkeypatch.setenv("SCREENSHOTONE_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("SCREENSHOTONE_RETRIES", "3")
    from garden_preview import capture_screenshots_for_urls

    png_bytes = __import__("base64").b64decode(_TINY_PNG_B64)
    bad = MagicMock()
    bad.ok = False
    bad.status_code = 500
    bad.json.return_value = {
        "error_message": "The API failed to serve your request.",
        "is_successful": False,
    }
    bad.text = "{}"

    good = MagicMock()
    good.ok = True
    good.headers = {"Content-Type": "image/png"}
    good.content = png_bytes

    mock_get.side_effect = [bad, good]

    out = capture_screenshots_for_urls(["https://example.com"])
    assert len(out) == 1
    assert out[0] == _TINY_PNG_B64
    assert mock_get.call_count == 2


@patch("garden_preview.requests.get")
def test_capture_screenshots_does_not_retry_on_400(mock_get, monkeypatch):
    monkeypatch.setenv("SCREENSHOTONE_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("SCREENSHOTONE_RETRIES", "5")
    from garden_preview import capture_screenshots_for_urls

    bad = MagicMock()
    bad.ok = False
    bad.status_code = 400
    bad.json.return_value = {"error_message": "bad request"}
    bad.text = "{}"
    mock_get.return_value = bad

    with pytest.raises(RuntimeError, match="HTTP 400"):
        capture_screenshots_for_urls(["https://example.com"])
    assert mock_get.call_count == 1


# --- integration: PNG images + JSON artifacts (real APIs) -------------------


@pytest.mark.integration
def test_live_pipeline_writes_png_and_json_artifacts(integration_gate):
    """
    End-to-end: ScreenshotOne → Gemini → Featherless; persist JSON traces and PNG
    screenshots (raster outputs are PNG, not JPEG).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sites = [
        {"url": "https://example.com", "criteria": ""},
        {
            "url": "https://example.org",
            "criteria": (
                "Focus on the main heading and primary navigation or links; "
                "describe distinct UI regions."
            ),
        },
    ]
    urls = [s["url"] for s in sites]

    result = run_garden_preview_pipeline(
        screenshots_b64=[],
        sites=sites,
        include_screenshots_b64=True,
    )

    images_b64 = result.pop("screenshots_b64", [])
    assert len(images_b64) == len(urls)

    shot_rel = _write_pngs_from_b64(
        OUTPUT_DIR / "screenshotone", urls, images_b64
    )
    gemini_rel = _write_pngs_from_b64(OUTPUT_DIR / "gemini", urls, images_b64)

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_instruction = gemini_vision_instruction_text(
        sites=sites,
        screenshot_count=len(images_b64),
    )
    featherless_msgs = featherless_preview_messages(
        sites=sites,
        gemini_context=result.get("gemini_raw"),
    )

    api_views = {
        "screenshotone": {
            "description": (
                "ScreenshotOne is called once per URL (GET /take). "
                "It returns image/png bytes; we store them under screenshotone/."
            ),
            "request": {
                "urls_in_order": urls,
                "per_url_criteria": sites,
                "env_note": "Viewport, timeout, etc. from SCREENSHOTONE_* env vars.",
            },
            "response_images_relative": shot_rel,
        },
        "gemini": {
            "description": (
                "Gemini receives a text instruction plus each PNG as an image part "
                "(same files as under gemini/)."
            ),
            "model": gemini_model,
            "text_instruction": gemini_instruction,
            "images_relative": gemini_rel,
            "structured_output_shape": (
                "JSON with `screenshots[]` (preview_html + optional preview_image) — "
                "see `gemini_raw` in last_response.json."
            ),
        },
        "featherless": {
            "description": (
                "Featherless chat/completions receives text-only messages (system + user). "
                "The user message embeds Gemini's JSON analysis as text."
            ),
            "messages": featherless_msgs,
            "structured_output_shape": (
                "JSON with `previews[]` (preview_html + optional preview_image) — "
                "see `featherless_raw` in last_response.json."
            ),
        },
    }
    api_views_path = OUTPUT_DIR / "api_views.json"
    api_views_path.write_text(json.dumps(api_views, indent=2), encoding="utf-8")

    json_path = OUTPUT_DIR / "last_response.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    previews = result.get("previews") or []
    for i, p in enumerate(previews):
        fh = p.get("featherless") or {}
        html_fragment = fh.get("preview_html") or ""
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

    print("\n========== GARDN LIVE PREVIEW ==========")
    print("screenshot_source:", result.get("screenshot_source"))
    print("previews count:", len(previews))
    for i, p in enumerate(previews):
        print(f"  [{i}] {p.get('title')!r} — {p.get('source_url')!r}")
        gm = p.get("gemini") or {}
        sm = gm.get("summary")
        if sm:
            print(f"       gemini summary: {str(sm)[:120]}...")
    print(f"Wrote: {json_path}")
    print(f"API trace: {api_views_path}")
    print(f"PNG artifacts: {OUTPUT_DIR / 'screenshotone'}, {OUTPUT_DIR / 'gemini'}")
    print(f"Open HTML files in: {OUTPUT_DIR}")
    print("========================================\n")

    # JSON outputs
    assert json_path.is_file()
    assert api_views_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded.get("screenshot_source") == "screenshotone"
    assert isinstance(loaded.get("sites"), list)
    assert len(loaded["sites"]) == len(sites)
    views_loaded = json.loads(api_views_path.read_text(encoding="utf-8"))
    assert "screenshotone" in views_loaded and "gemini" in views_loaded
    assert "featherless" in views_loaded

    # PNG image outputs (screenshots are PNG from ScreenshotOne)
    assert (OUTPUT_DIR / "screenshotone" / Path(shot_rel[0]).name).is_file()
    assert (OUTPUT_DIR / "gemini" / Path(gemini_rel[0]).name).is_file()
    for rel in shot_rel:
        _assert_png_file(OUTPUT_DIR / rel)
    for rel in gemini_rel:
        _assert_png_file(OUTPUT_DIR / rel)

    # Response shape
    assert len(previews) >= 1
    assert len(previews) == len(sites)
    for row in previews:
        assert "gemini" in row and "featherless" in row
        assert "preview_html" in row["gemini"]
        assert "preview_html" in row["featherless"]
