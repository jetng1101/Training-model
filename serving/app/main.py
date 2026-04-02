from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile

from serving.app.config import settings
from serving.app.model import GradingModel
from serving.app.schemas import HealthResponse, PredictionResponse, PredictionScore

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

model: GradingModel | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup, release on shutdown."""
    global model
    model = GradingModel(settings)
    yield
    model = None


app = FastAPI(
    title="Grading Service",
    description="Image classification API for product condition grading.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def check_health() -> HealthResponse:
    """Check if the service and model are ready."""
    return HealthResponse(
        status="ok" if model and model.is_loaded else "unavailable",
        model_loaded=model is not None and model.is_loaded,
        device=settings.device,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """Upload an image and get the predicted grade/class."""
    if model is None or not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place checkpoint at: " + settings.checkpoint_path)
    _validate_upload(file)
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")
    result = model.predict(image_bytes)
    top_scores = sorted(
        result.all_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return PredictionResponse(
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        scores=[
            PredictionScore(label=label, confidence=score)
            for label, score in top_scores
        ],
    )


def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )
