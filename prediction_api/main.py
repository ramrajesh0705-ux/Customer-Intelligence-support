from fastapi import FastAPI, HTTPException

from .model_service import ModelService
from .schemas import PredictionResponse, TicketRequest


app = FastAPI(title="Customer Intelligence Prediction API", version="1.0.0")
model_service = ModelService()


@app.get("/health")
def health() -> dict:
    return {"service": "prediction-api", **model_service.status()}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(ticket: TicketRequest) -> PredictionResponse:
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail={"message": "No prediction models are available", "errors": model_service.errors})
    try:
        return model_service.predict(ticket)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc