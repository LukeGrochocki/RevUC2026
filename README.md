Hi this is our project :D

## Run the API locally

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Server: `http://localhost:8000` — OpenAPI docs at `http://localhost:8000/docs`.

**Environment (`.env`):**

Copy `.env.example` to `.env` and fill in secrets. Never commit `.env`. If keys were ever exposed (e.g. in chat or a public repo), rotate them in the Featherless and Google AI dashboards.

- `FEATHERLESS_API_KEY` — required for HTML previews.
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` — required if you send `screenshots` (Gemini vision pass).
- Optional: `GEMINI_MODEL` (default `gemini-2.0-flash`), `FEATHERLESS_MODEL`, `CORS_ORIGINS` (comma-separated; default `*`).

## Frontend ↔ backend

`POST /api/garden-preview` with JSON:

- `urls` — list of page URLs.
- `user_notes` — what to pull out (e.g. a `<nav>`, a button, layout notes).
- `screenshots` — optional list of base64 strings or `data:image/...;base64,...` images; index `i` pairs with `urls[i]` when you have one shot per URL.
- `include_model_debug` — optional; if `true`, response includes raw Gemini/Featherless JSON.

Response `previews` is a list of **preview containers**: each item has `title`, `source_url`, `preview_html` (iframe-safe fragment), `featherless_notes`, `visual_summary`, and `screenshot_analysis` when Gemini ran.
