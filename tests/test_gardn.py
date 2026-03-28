"""API and pipeline tests with mocked LLM calls (no real API keys required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import httpx

from garden_preview import _featherless_message_content, decode_base64_image, featherless_chat
from main import app, _cors_allow_origins

# Minimal valid 1×1 PNG (transparent pixel)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_GEMINI_MIN = {
    "screenshots": [
        {
            "url_index": 0,
            "url": "https://example.com",
            "visual_description": "mock visual",
            "requested_elements_focus": "mock focus",
        }
    ]
}


@pytest.fixture
def client():
    return TestClient(app)


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
    """Vision path uses `google-genai` (Client + Part.from_bytes), not deprecated google-generativeai."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.candidates = [MagicMock()]
    mock_resp.text = json.dumps(
        {
            "screenshots": [
                {
                    "url_index": 0,
                    "url": "https://example.com",
                    "visual_description": "x",
                    "requested_elements_focus": "y",
                }
            ]
        }
    )
    mock_inst = MagicMock()
    mock_inst.models.generate_content.return_value = mock_resp
    mock_client_cls.return_value = mock_inst

    from garden_preview import gemini_analyze_screenshots

    out = gemini_analyze_screenshots(
        urls=["https://example.com"],
        user_notes="note",
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
                    "preview_html": '<div style="padding:8px">nav</div>',
                    "notes": "mock preview",
                }
            ]
        }
    )

    r = client.post(
        "/api/garden-preview",
        json={
            "urls": ["https://example.com"],
            "user_notes": "the nav bar",
            "screenshots": [],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["screenshot_source"] == "screenshotone"
    assert "previews" in body
    assert len(body["previews"]) == 1
    assert body["previews"][0]["title"] == "Navigation"
    assert body["previews"][0]["visual_summary"] == "mock visual"
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
                    "preview_html": "<div>nav</div>",
                    "notes": "mock",
                }
            ]
        }
    )

    r = client.post(
        "/api/garden-preview",
        json={
            "urls": ["https://example.com"],
            "user_notes": "nav",
            "screenshots": [_TINY_PNG_B64],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["screenshot_source"] == "client"
    assert body["previews"][0]["visual_summary"] == "mock visual"
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
                    "preview_html": "<div/>",
                    "notes": "n",
                }
            ]
        }
    )
    r = client.post(
        "/api/garden-preview",
        json={
            "urls": ["https://a.test"],
            "user_notes": "",
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
        json={"urls": ["https://example.com"], "user_notes": "", "screenshots": []},
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
