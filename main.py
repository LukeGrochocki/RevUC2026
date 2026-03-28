import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from garden_preview import run_garden_preview_pipeline

load_dotenv()

app = FastAPI(title="Gardn API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GardenPreviewRequest(BaseModel):
    """Payload from the frontend: URLs, what to emphasize, optional screenshot(s) per URL."""

    urls: list[str] = Field(
        ...,
        min_length=1,
        description="Page URLs the user is pulling from (order matches screenshots).",
    )
    user_notes: str = Field(
        default="",
        description="What they want (e.g. the <nav>, a specific button, hero layout).",
    )
    screenshots: list[str] = Field(
        default_factory=list,
        description="Base64 or data-URL images; index i pairs with urls[i] when lengths match.",
    )
    include_model_debug: bool = Field(
        default=False,
        description="If true, include gemini_raw and featherless_raw in the response.",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/garden-preview")
def garden_preview(body: GardenPreviewRequest):
    """
    Runs Gemini on screenshots (if any), then Featherless to produce HTML preview fragments.
    Returns rows the UI can render in iframe / shadow DOM containers.
    """
    try:
        out = run_garden_preview_pipeline(
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
