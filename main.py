import os
import warnings

# Upstream google-genai uses typing aliases deprecated on Python 3.14+ (noise in logs).
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"google\.genai\.types",
)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from garden_preview import run_garden_preview_pipeline

load_dotenv()


def _cors_allow_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    if raw.strip() == "" or raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="Gardn API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GardenSite(BaseModel):
    """One URL and optional criteria; empty criteria means analyze overall page style."""

    url: str = Field(..., description="Page URL to capture (ScreenshotOne or client image).")
    criteria: str = Field(
        default="",
        description=(
            "What to analyze for this URL (e.g. hero, nav). Leave empty for whole-site style."
        ),
    )


class GardenPreviewRequest(BaseModel):
    """Per-site criteria; screenshots optional override when length matches URLs."""

    sites: list[GardenSite] | None = Field(
        default=None,
        description="URLs with optional criteria each; preferred over legacy urls/user_notes.",
    )
    urls: list[str] | None = Field(
        default=None,
        description="Legacy: same criteria (user_notes) applied to every URL.",
    )
    user_notes: str = Field(
        default="",
        description="Legacy: applied to each URL when `sites` is omitted.",
    )
    screenshots: list[str] = Field(
        default_factory=list,
        description=(
            "Optional. If len(screenshots) matches the number of sites/URLs, these "
            "base64/data-URL images are used instead of ScreenshotOne (testing/overrides). "
            "Otherwise leave empty to capture each URL server-side."
        ),
    )
    include_model_debug: bool = Field(
        default=False,
        description="If true, include gemini_raw and featherless_raw in the response.",
    )

    @model_validator(mode="after")
    def _require_sites_or_urls(self):
        if self.sites and len(self.sites) > 0:
            return self
        if self.urls and len(self.urls) > 0:
            return self
        raise ValueError("Provide non-empty `sites` or legacy `urls`.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/garden-preview")
def garden_preview(body: GardenPreviewRequest):
    """
    Captures each URL with ScreenshotOne (unless client sends matching-length screenshots),
    runs Gemini per-site criteria on those images, then Featherless for HTML previews.
    """
    try:
        if body.sites and len(body.sites) > 0:
            site_dicts = [s.model_dump() for s in body.sites]
        else:
            site_dicts = None
        out = run_garden_preview_pipeline(
            sites=site_dicts,
            urls=body.urls,
            user_notes=body.user_notes,
            screenshots_b64=body.screenshots,
        )
        if not body.include_model_debug:
            out.pop("gemini_raw", None)
            out.pop("featherless_raw", None)
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
