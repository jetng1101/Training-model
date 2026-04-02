from pydantic import BaseModel, Field


class PredictionScore(BaseModel):
    """Score for a single class label."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response for a single image prediction."""

    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: list[PredictionScore]


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str
    model_loaded: bool
    device: str
