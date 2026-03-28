"""Orchestrates Gemini (screenshot vision) + Featherless (HTML previews) for Gardn."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import requests
from requests.exceptions import RequestException

SCREENSHOTONE_TAKE_URL = "https://api.screenshotone.com/take"


def _screenshotone_error_message(r: requests.Response) -> str:
    try:
        data = r.json()
        if isinstance(data, dict):
            em = data.get("error_message")
            if em is not None:
                return str(em)[:500]
            err = data.get("error")
            if isinstance(err, dict) and err.get("message") is not None:
                return str(err["message"])[:500]
    except Exception:
        pass
    return (r.text or "")[:500]


def _screenshotone_status_retryable(status: int) -> bool:
    return status == 429 or status >= 500


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


# When criteria for a URL is empty, Gemini/Featherless use this analysis target.
DEFAULT_SITE_CRITERIA = (
    "Analyze the overall visual style of the entire page: typography, color palette, "
    "spacing, and layout of major regions (header, content, footer as visible)."
)


def effective_criteria(raw: str) -> str:
    """Return criteria to send to models; empty/whitespace → whole-site default."""
    t = (raw or "").strip()
    return t if t else DEFAULT_SITE_CRITERIA


def normalize_sites(
    sites: list[dict[str, str]] | None,
    *,
    urls: list[str] | None = None,
    user_notes: str = "",
) -> list[dict[str, str]]:
    """
    Build a list of {url, criteria} dicts. Prefer ``sites``; otherwise legacy ``urls``
    with the same ``user_notes`` applied to each URL.
    """
    if sites is not None and len(sites) > 0:
        return [{"url": s["url"], "criteria": s.get("criteria", "")} for s in sites]
    if urls:
        return [{"url": u, "criteria": user_notes} for u in urls]
    raise ValueError("Provide non-empty `sites` or `urls`.")


def render_html_fragment_to_png_base64(html_fragment: str) -> str | None:
    """
    Rasterize a small HTML fragment to PNG (base64, no data-URL prefix).
    Uses headless Chrome via html2image when available; returns None if disabled
    or rendering fails (no Chromium, etc.).
    """
    if os.getenv("GARDN_DISABLE_HTML_PREVIEW", "").lower() in ("1", "true", "yes"):
        return None
    frag = (html_fragment or "").strip()
    if not frag:
        return None
    try:
        from html2image import Html2Image
    except ImportError:
        return None
    full = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
        "<style>html,body{margin:0;padding:12px;box-sizing:border-box;"
        "font-family:system-ui,sans-serif;background:#fff;}</style></head><body>"
        f"{frag}</body></html>"
    )
    w = int(os.getenv("GARDN_PREVIEW_WIDTH", "800"))
    h = int(os.getenv("GARDN_PREVIEW_HEIGHT", "600"))
    try:
        with tempfile.TemporaryDirectory() as td:
            hti = Html2Image(output_path=td, size=(w, h))
            hti.screenshot(html_str=full, save_as="garden_preview.png")
            data = (Path(td) / "garden_preview.png").read_bytes()
            return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _coerce_preview_image_b64(
    html: str, explicit: str | None
) -> str | None:
    if explicit and str(explicit).strip():
        s = str(explicit).strip()
        try:
            decode_base64_image(s)
        except Exception:
            pass
        else:
            return s
    return render_html_fragment_to_png_base64(html)


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
    max_attempts = max(1, int(os.getenv("SCREENSHOTONE_RETRIES", "4")))
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

        for attempt in range(max_attempts):
            try:
                r = requests.get(
                    SCREENSHOTONE_TAKE_URL, params=params, timeout=timeout
                )
            except RequestException as e:
                if attempt + 1 >= max_attempts:
                    raise RuntimeError(
                        f"ScreenshotOne request failed for {url!r} after "
                        f"{max_attempts} attempt(s): {e!s}"
                    ) from e
                time.sleep(min(2.0**attempt, 12.0))
                continue

            if r.ok:
                ct = (r.headers.get("Content-Type") or "").lower()
                if "image" not in ct and "octet-stream" not in ct:
                    raise RuntimeError(
                        f"ScreenshotOne returned non-image for {url!r} "
                        f"(Content-Type={ct!r}): {r.text[:400]}"
                    )
                out.append(base64.b64encode(r.content).decode("ascii"))
                break

            msg = _screenshotone_error_message(r)
            if _screenshotone_status_retryable(r.status_code) and attempt + 1 < max_attempts:
                time.sleep(min(2.0**attempt, 12.0))
                continue
            raise RuntimeError(
                f"ScreenshotOne failed for {url!r}: HTTP {r.status_code} {msg}"
            )

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
    sites: list[dict[str, str]],
    screenshots_b64: list[str],
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Vision pass: for each screenshot, apply that URL's criteria and return HTML plus
    optional preview image (filled server-side if omitted).
    Expects ``screenshots_b64`` aligned by index with ``sites``.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY for screenshot analysis."
        )

    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError as GenaiClientError

    client = genai.Client(api_key=api_key)
    # gemini-2.0-flash is not available to new API keys; default to a current model.
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    parts: list[Any] = []
    intro = gemini_vision_instruction_text(
        sites=sites,
        screenshot_count=len(screenshots_b64),
    )
    parts.append(intro)

    for i, b64 in enumerate(screenshots_b64):
        raw = decode_base64_image(b64)
        parts.append(f"[Screenshot index {i}]")
        parts.append(
            types.Part.from_bytes(data=raw, mime_type="image/png"),
        )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=parts,
        )
    except GenaiClientError as e:
        raise RuntimeError(
            f"Gemini API error: {e}. "
            "Set GEMINI_MODEL in .env to a model your key supports "
            "(e.g. gemini-2.5-flash or gemini-2.5-pro)."
        ) from e

    if not getattr(response, "candidates", None):
        raise RuntimeError("Gemini returned no candidates (blocked or empty).")
    text = _gemini_response_text(response)
    if not text.strip():
        raise RuntimeError("Gemini returned empty text.")
    parsed = parse_model_json(text)
    _attach_gemini_preview_images(parsed)
    return parsed


def _attach_gemini_preview_images(gemini_context: dict[str, Any]) -> None:
    for s in gemini_context.get("screenshots") or []:
        if not isinstance(s, dict):
            continue
        html = s.get("preview_html") or ""
        img = s.get("preview_image_png_base64")
        b64 = _coerce_preview_image_b64(html, img if isinstance(img, str) else None)
        if b64:
            s["preview_image_png_base64"] = b64


def gemini_vision_instruction_text(
    *,
    sites: list[dict[str, str]],
    screenshot_count: int,
) -> str:
    """Text instruction Gemini receives before the screenshot image parts (for tracing/debug)."""
    site_rows: list[dict[str, Any]] = []
    for i, site in enumerate(sites):
        u = site.get("url", "")
        c_raw = site.get("criteria", "")
        site_rows.append(
            {
                "index": i,
                "url": u,
                "criteria": c_raw,
                "effective_analysis_target": effective_criteria(c_raw),
            }
        )
    sites_block = json.dumps(site_rows, indent=2)

    return f"""You are helping users remix pieces of websites into a "garden" collage.

Sites (one screenshot per site, same order — index i matches screenshot i). If criteria is empty for a URL, analyze the overall visual style of the whole page (see effective_analysis_target in the JSON above).
{sites_block}

There are {screenshot_count} screenshot(s) in order. Each image is the page for the URL at the same index.

For EACH screenshot in order, produce a small self-contained HTML fragment (inline CSS only, no external assets) that reflects what was asked: either the user's criteria for that URL, or the overall visual style of the whole page if criteria was left empty. Also include a short summary sentence.

Respond with ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "screenshots": [
    {{
      "url_index": <int, 0-based index, should match screenshot order>,
      "url": "<string from the list>",
      "criteria": "<user's criteria string for this URL, or empty string>",
      "effective_analysis_target": "<the analysis target you used (verbatim from effective_analysis_target in the list or the default whole-page style)>",
      "summary": "<one sentence>",
      "preview_html": "<div style=\\"...\\">...</div>",
      "preview_image_png_base64": null
    }}
  ]
}}

Set preview_image_png_base64 to JSON null; the server may fill a raster preview. Focus on accurate, compact preview_html."""


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


def _featherless_collect_sse_assistant_text(response: httpx.Response) -> str:
    """Accumulate assistant text from an OpenAI-style chat completions SSE stream."""
    parts: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            raw = line[5:].strip()
        else:
            continue
        if raw == "[DONE]":
            break
        try:
            chunk: Any = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            c = delta.get("content")
            if isinstance(c, str) and c:
                parts.append(c)
            elif isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text")
                        if isinstance(t, str):
                            parts.append(t)
    out = "".join(parts)
    if not out.strip():
        raise RuntimeError(
            "Featherless streaming returned no assistant text (empty SSE or parse miss)."
        )
    return out


def featherless_chat(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    key = os.getenv("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError("FEATHERLESS_API_KEY is not set.")

    model = model or os.getenv(
        "FEATHERLESS_MODEL", "moonshotai/Kimi-K2.5"
    )
    # Long generations + chunked responses can drop mid-stream; retry transient errors.
    req_timeout = timeout if timeout is not None else int(
        os.getenv("FEATHERLESS_TIMEOUT", "300")
    )
    max_attempts = max(1, int(os.getenv("FEATHERLESS_RETRIES", "5")))
    url = "https://api.featherless.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        # Avoid pooled keep-alive connections that some proxies truncate on long bodies.
        "Connection": "close",
    }
    # Cap completion size — very long JSON+HTML bodies often hit incomplete chunked reads upstream.
    max_tokens = int(os.getenv("FEATHERLESS_MAX_TOKENS", "4096"))
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    # SSE streaming uses small incremental chunks; avoids one huge non-streaming chunked body.
    use_stream = os.getenv("FEATHERLESS_STREAM", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    timeout = httpx.Timeout(req_timeout)

    for attempt in range(max_attempts):
        try:
            if use_stream:
                stream_payload = {**payload, "stream": True}
                with httpx.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=stream_payload,
                    timeout=timeout,
                ) as r:
                    if not r.is_success:
                        err_body = r.read().decode("utf-8", errors="replace")[:800]
                        raise RuntimeError(
                            f"Featherless HTTP {r.status_code}: {err_body}"
                        )
                    return _featherless_collect_sse_assistant_text(r)
            r = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if not r.is_success:
                raise RuntimeError(f"Featherless HTTP {r.status_code}: {r.text}")
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                body_preview = (r.text or "")[:500]
                raise RuntimeError(
                    f"Featherless returned non-JSON body: {body_preview}"
                ) from e
            return _featherless_message_content(data)
        except httpx.TransportError as e:
            if attempt + 1 >= max_attempts:
                raise RuntimeError(
                    f"Featherless connection failed after {max_attempts} attempt(s): {e!s}. "
                    "This is often a dropped chunked response; try again or increase "
                    "FEATHERLESS_TIMEOUT / FEATHERLESS_RETRIES."
                ) from e
            time.sleep(min(2.0**attempt, 15.0))


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


def featherless_preview_messages(
    *,
    sites: list[dict[str, str]],
    gemini_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Chat messages sent to Featherless (same as `featherless_build_previews` uses)."""
    if gemini_context:
        gemini_blob = json.dumps(gemini_context, separators=(",", ":"))
        _max = int(os.getenv("FEATHERLESS_GEMINI_CONTEXT_CHARS", "20000"))
        if len(gemini_blob) > _max:
            gemini_blob = gemini_blob[:_max] + "\n…(truncated for prompt size)"
    else:
        gemini_blob = "(no screenshot analysis — rely on sites only)"
    system = """You are a front-end expert building small preview snippets for a product called Gardn.
Each preview is one self-contained HTML fragment: a single outer element using inline CSS only (no external assets unless HTTPS URLs from the user's context).
Output ONLY valid JSON, no markdown fences, no commentary outside JSON."""

    site_rows = []
    for i, site in enumerate(sites):
        c_raw = site.get("criteria", "")
        site_rows.append(
            {
                "index": i,
                "url": site.get("url", ""),
                "criteria": c_raw,
                "effective_analysis_target": effective_criteria(c_raw),
            }
        )

    user = f"""Sites (one preview per site, same order). If criteria is empty, model the overall visual style of the whole page (see effective_analysis_target).
{json.dumps(site_rows, indent=2)}

Screenshot / vision analysis from Gemini (HTML fragments + metadata per URL):
{gemini_blob}

Return JSON with this exact shape:
{{
  "previews": [
    {{
      "title": "short label for this URL's preview",
      "source_url": "must match the site URL at the same index",
      "criteria": "<same criteria string as input for that site, or empty>",
      "preview_html": "<div style=\\"...\\">...</div>",
      "preview_image_png_base64": null,
      "notes": "one sentence on what this preview shows"
    }}
  ]
}}

There must be exactly {len(sites)} preview object(s), in the same order as the sites list.
Set preview_image_png_base64 to JSON null; the server may fill a raster preview.
Keep each preview_html compact (minimal inline CSS, no long copy)."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def featherless_build_previews(
    *,
    sites: list[dict[str, str]],
    gemini_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate self-contained HTML preview snippets per site."""
    messages = featherless_preview_messages(
        sites=sites,
        gemini_context=gemini_context,
    )
    raw = featherless_chat(messages=messages)
    parsed = parse_model_json(raw)
    _attach_featherless_preview_images(parsed)
    return parsed


def _attach_featherless_preview_images(featherless_json: dict[str, Any]) -> None:
    for p in featherless_json.get("previews") or []:
        if not isinstance(p, dict):
            continue
        html = p.get("preview_html") or ""
        img = p.get("preview_image_png_base64")
        b64 = _coerce_preview_image_b64(html, img if isinstance(img, str) else None)
        if b64:
            p["preview_image_png_base64"] = b64


def merge_preview_response(
    gemini: dict[str, Any] | None,
    featherless: dict[str, Any],
    sites: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    One row per site: criteria, Gemini HTML+image, Featherless HTML+image, plus notes.
    """
    shots = (gemini or {}).get("screenshots") or []
    previews = featherless.get("previews") or []
    out: list[dict[str, Any]] = []

    for i, site in enumerate(sites):
        url = site.get("url", "")
        crit = site.get("criteria", "")
        p = previews[i] if i < len(previews) else {}
        g = None
        for s in shots:
            if isinstance(s, dict) and s.get("url") == url:
                g = s
                break
        if g is None and i < len(shots):
            g = shots[i]
        if not isinstance(g, dict):
            g = {}
        if not isinstance(p, dict):
            p = {}

        gh = g.get("preview_html") or ""
        ph = p.get("preview_html") or ""
        row = {
            "title": p.get("title", f"Site {i + 1}"),
            "source_url": p.get("source_url") or url,
            "criteria": p.get("criteria", crit),
            "gemini": {
                "preview_html": gh,
                "preview_image_b64": g.get("preview_image_png_base64"),
                "summary": g.get("summary"),
                "effective_analysis_target": g.get("effective_analysis_target"),
            },
            "featherless": {
                "preview_html": ph,
                "preview_image_b64": p.get("preview_image_png_base64"),
            },
            "featherless_notes": p.get("notes", ""),
        }
        out.append(row)

    return out


def run_garden_preview_pipeline(
    *,
    screenshots_b64: list[str],
    sites: list[dict[str, str]] | None = None,
    urls: list[str] | None = None,
    user_notes: str = "",
    include_screenshots_b64: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline: resolve screenshots (ScreenshotOne per URL, or client images if
    lengths match), Gemini (per-site criteria), then Featherless HTML previews.

    Pass ``sites`` as ``[{ "url", "criteria" }, ...]`` or legacy ``urls`` + ``user_notes``.

    If ``include_screenshots_b64`` is True, the returned dict includes ``screenshots_b64``
    (same PNGs passed to Gemini) for debugging or integration artifacts.
    """
    resolved = normalize_sites(sites, urls=urls, user_notes=user_notes)
    url_list = [s["url"] for s in resolved]

    images_b64, screenshot_source = resolve_screenshot_b64_list(url_list, screenshots_b64)

    gemini_context = gemini_analyze_screenshots(
        sites=resolved,
        screenshots_b64=images_b64,
    )

    featherless_json = featherless_build_previews(
        sites=resolved,
        gemini_context=gemini_context,
    )

    merged = merge_preview_response(gemini_context, featherless_json, resolved)

    out: dict[str, Any] = {
        "sites": resolved,
        "previews": merged,
        "screenshot_source": screenshot_source,
        "gemini_raw": gemini_context,
        "featherless_raw": featherless_json,
    }
    if include_screenshots_b64:
        out["screenshots_b64"] = images_b64
    return out
