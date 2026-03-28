Hi this is our project :D

## Run the API locally

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Server: `http://localhost:8000` — OpenAPI docs at `http://localhost:8000/docs`.

Run tests (mocks LLMs; no API keys needed): `.venv/bin/pytest -v`

**Live demo (real APIs, costs quota):** fill `.env`, then either:

- `.venv/bin/python scripts/demo_preview.py` — prints a short summary and writes `output/demo_last_response.json` plus `output/demo_preview_*.html` (open the HTML in a browser).
- `GARDN_INTEGRATION=1 .venv/bin/pytest tests/integration -v -s` — same idea; artifacts under `tests/integration/output/`.

**Environment (`.env`):**

Copy `.env.example` to `.env` and fill in secrets. Never commit `.env`. If keys were ever exposed (e.g. in chat or a public repo), rotate them in the Featherless and Google AI dashboards.

- `FEATHERLESS_API_KEY` — required for HTML previews.
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` — required for Gemini (vision of captured screenshots). The app uses the **`google-genai`** Python SDK (not the deprecated `google-generativeai` package).
- `SCREENSHOTONE_ACCESS_KEY` — required unless the client sends `screenshots` with the same length as `urls` (override).
- Optional: `GEMINI_MODEL` (default `gemini-2.5-flash`), `FEATHERLESS_MODEL`, `FEATHERLESS_TIMEOUT` (default `300` seconds), `FEATHERLESS_RETRIES` (default `3`, for transient chunked-stream drops), `CORS_ORIGINS` (comma-separated; default `*`), `SCREENSHOTONE_VIEWPORT_WIDTH` / `HEIGHT`, `SCREENSHOTONE_TIMEOUT`, `SCREENSHOTONE_FULL_PAGE`.

## Frontend ↔ backend

`POST /api/garden-preview` with JSON:

- `urls` — list of page URLs (server captures each with **ScreenshotOne** unless you override with `screenshots`).
- `user_notes` — what to pull out (e.g. a `<nav>`, a button, layout notes).
- `screenshots` — optional; only used when `len(screenshots) === len(urls)` (base64 or data URLs). Otherwise leave `[]` for automatic capture.
- `include_model_debug` — optional; if `true`, response includes raw Gemini/Featherless JSON.

Response includes `screenshot_source` (`"screenshotone"` or `"client"`) and `previews`: each item has `title`, `source_url`, `preview_html`, `featherless_notes`, `visual_summary`, and `screenshot_analysis` from Gemini.
