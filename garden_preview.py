"""Orchestrates Gemini (screenshot vision) + Featherless (HTML previews) for Gardn."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import requests

SCREENSHOTONE_TAKE_URL = "https://api.screenshotone.com/take"


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def parse_model_json(content: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating ```json fences."""
    cleaned = _strip_json_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        snippet = cleaned[:800] + ("…" if len(cleaned) > 800 else "")
        raise RuntimeError(f"Model output was not valid JSON: {e}\n---\n{snippet}") from e


def decode_base64_image(data: str) -> bytes:
    """Decode raw base64 or a data URL (data:image/png;base64,...)."""
    s = data.strip()
    if s.startswith("data:"):
        if "," not in s:
            raise ValueError("Invalid data URL: missing comma")
        s = s.split(",", 1)[1]
    return base64.b64decode(s, validate=True)


def capture_screenshots_for_urls(urls: list[str]) -> list[str]:
    """
    Fetch a PNG screenshot per URL via ScreenshotOne.
    Returns raw base64-encoded image strings (no data: prefix), same order as urls.
    """
    access = os.getenv("SCREENSHOTONE_ACCESS_KEY")
    if not access:
        raise RuntimeError(
            "Set SCREENSHOTONE_ACCESS_KEY to capture screenshots of URLs "
            "(or pass client screenshots with the same length as urls)."
        )

    timeout = int(os.getenv("SCREENSHOTONE_TIMEOUT", "90"))
    viewport_w = int(os.getenv("SCREENSHOTONE_VIEWPORT_WIDTH", "1280"))
    viewport_h = int(os.getenv("SCREENSHOTONE_VIEWPORT_HEIGHT", "720"))
    out: list[str] = []

    for url in urls:
        params: dict[str, Any] = {
            "access_key": access,
            "url": url,
            "format": "png",
            "viewport_width": viewport_w,
            "viewport_height": viewport_h,
        }
        if os.getenv("SCREENSHOTONE_FULL_PAGE", "").lower() in ("1", "true", "yes"):
            params["full_page"] = "true"

        r = requests.get(SCREENSHOTONE_TAKE_URL, params=params, timeout=timeout)
        if not r.ok:
            try:
                err = r.json().get("error") or {}
                msg = err.get("message", r.text[:300])
            except Exception:
                msg = r.text[:300]
            raise RuntimeError(
                f"ScreenshotOne failed for {url!r}: HTTP {r.status_code} {msg}"
            )

        ct = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ct and "octet-stream" not in ct:
            raise RuntimeError(
                f"ScreenshotOne returned non-image for {url!r} (Content-Type={ct!r}): "
                f"{r.text[:400]}"
            )

        out.append(base64.b64encode(r.content).decode("ascii"))

    return out


def resolve_screenshot_b64_list(
    urls: list[str],
    client_screenshots_b64: list[str],
) -> tuple[list[str], str]:
    """
    Prefer client-provided images when count matches urls; otherwise ScreenshotOne.
    Returns (base64_strings, source) where source is 'client' or 'screenshotone'.
    """
    if client_screenshots_b64 and len(client_screenshots_b64) == len(urls):
        return client_screenshots_b64, "client"
    return capture_screenshots_for_urls(urls), "screenshotone"


def gemini_analyze_screenshots(
    *,
    urls: list[str],
    user_notes: str,
    screenshots_b64: list[str],
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Vision pass: describe each screenshot in context of URLs and user intent.
    Expects screenshots_b64 aligned by index with urls (same length ideal).
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY for screenshot analysis."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    parts: list[Any] = []
    intro = f"""You are helping users remix pieces of websites into a "garden" collage.

URLs provided (in order, index starting at 0):
{json.dumps(urls, indent=2)}

What the user wants to capture or emphasize (specific tags, components, or vibes):
{user_notes or "(none)"}

There are {len(screenshots_b64)} screenshot(s) in order. Each image corresponds to the URL at the same index when counts match; otherwise describe each image and relate it to the closest URL by context.

For EACH screenshot in order, respond with ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "screenshots": [
    {{
      "url_index": <int, 0-based index of the URL this shot best matches>,
      "url": "<string, copy from list or best guess>",
      "visual_description": "<colors, typography, spacing, notable UI regions>",
      "requested_elements_focus": "<how the user's requested parts appear in this shot>"
    }}
  ]
}}
"""
    parts.append(intro)

    for i, b64 in enumerate(screenshots_b64):
        raw = decode_base64_image(b64)
        parts.append(f"[Screenshot index {i}]")
        parts.append(
            types.Part.from_bytes(data=raw, mime_type="image/png"),
        )

    response = client.models.generate_content(
        model=model_name,
        contents=parts,
    )
    if not getattr(response, "candidates", None):
        raise RuntimeError("Gemini returned no candidates (blocked or empty).")
    text = _gemini_response_text(response)
    if not text.strip():
        raise RuntimeError("Gemini returned empty text.")
    return parse_model_json(text)


def _gemini_response_text(response: Any) -> str:
    try:
        t = response.text
        if t:
            return t
    except Exception:
        pass
    chunks: list[str] = []
    for cand in response.candidates or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)
    return "".join(chunks)


def featherless_chat(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    timeout: int = 180,
) -> str:
    key = os.getenv("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError("FEATHERLESS_API_KEY is not set.")

    model = model or os.getenv(
        "FEATHERLESS_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
    )
    r = requests.post(
        "https://api.featherless.ai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        json={"model": model, "messages": messages},
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"Featherless HTTP {r.status_code}: {r.text}")
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Featherless returned non-JSON body: {r.text[:500]}") from e
    return _featherless_message_content(data)


def _featherless_message_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Featherless response JSON was not an object (got {type(data).__name__})."
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise RuntimeError(
            "Featherless response missing or empty 'choices'; "
            f"keys present: {list(data.keys())!r}."
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("Featherless response choices[0] is not an object.")
    msg = first.get("message")
    if not isinstance(msg, dict):
        raise RuntimeError(
            "Featherless response missing a 'message' object on the first choice."
        )
    content = msg.get("content")
    if content is None:
        raise RuntimeError(
            "Featherless response missing 'content' on the assistant message."
        )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some APIs return structured content parts
        parts_out: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts_out.append(t)
            elif isinstance(block, str):
                parts_out.append(block)
        if parts_out:
            return "".join(parts_out)
    return str(content)


def featherless_build_previews(
    *,
    urls: list[str],
    user_notes: str,
    gemini_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate self-contained HTML preview snippets per requested part."""
    gemini_blob = (
        json.dumps(gemini_context, indent=2)
        if gemini_context
        else "(no screenshot analysis — rely on URLs and user notes only)"
    )
    system = """You are a front-end expert building small preview snippets for a product called Gardn.
Each preview is one self-contained HTML fragment: a single outer element using inline CSS only (no external assets unless HTTPS URLs from the user's context).
Output ONLY valid JSON, no markdown fences, no commentary outside JSON."""

    user = f"""URLs:
{json.dumps(urls, indent=2)}

User notes (what to extract or prioritize, e.g. <nav>, a specific button, hero):
{user_notes or "(none)"}

Screenshot / vision analysis (from Gemini):
{gemini_blob}

Return JSON with this exact shape:
{{
  "previews": [
    {{
      "title": "short label e.g. Navigation",
      "source_url": "must be one of the URLs above or the closest match",
      "preview_html": "<div style=\\"...\\">...</div>",
      "notes": "one sentence on what this preview shows"
    }}
  ]
}}

Create one entry per distinct part the user asked for; if vague, pick up to {min(5, max(1, len(urls)))} clear UI regions."""

    raw = featherless_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return parse_model_json(raw)


def merge_preview_response(
    gemini: dict[str, Any] | None,
    featherless: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build frontend-friendly rows: each row pairs Featherless HTML with Gemini visual text when possible.
    """
    shots = (gemini or {}).get("screenshots") or []
    previews = featherless.get("previews") or []
    out: list[dict[str, Any]] = []

    for i, p in enumerate(previews):
        row = {
            "title": p.get("title", f"Part {i + 1}"),
            "source_url": p.get("source_url"),
            "preview_html": p.get("preview_html", ""),
            "featherless_notes": p.get("notes", ""),
            "visual_summary": None,
            "screenshot_analysis": None,
        }
        url = p.get("source_url") or ""
        # Match Gemini entry by url or by index
        g = None
        for s in shots:
            if s.get("url") == url:
                g = s
                break
        if g is None and i < len(shots):
            g = shots[i]
        if isinstance(g, dict):
            row["visual_summary"] = g.get("visual_description")
            row["screenshot_analysis"] = {
                "url_index": g.get("url_index"),
                "requested_elements_focus": g.get("requested_elements_focus"),
            }
        out.append(row)

    return out


def run_garden_preview_pipeline(
    *,
    urls: list[str],
    user_notes: str,
    screenshots_b64: list[str],
) -> dict[str, Any]:
    """
    Full pipeline: resolve screenshots (ScreenshotOne per URL, or client images if
    lengths match), Gemini vision pass, then Featherless HTML previews.
    """
    images_b64, screenshot_source = resolve_screenshot_b64_list(urls, screenshots_b64)

    gemini_context = gemini_analyze_screenshots(
        urls=urls,
        user_notes=user_notes,
        screenshots_b64=images_b64,
    )

    featherless_json = featherless_build_previews(
        urls=urls,
        user_notes=user_notes,
        gemini_context=gemini_context,
    )

    merged = merge_preview_response(gemini_context, featherless_json)

    return {
        "previews": merged,
        "screenshot_source": screenshot_source,
        "gemini_raw": gemini_context,
        "featherless_raw": featherless_json,
    }
