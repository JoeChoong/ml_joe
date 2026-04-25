import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="ML Scoring API", version="1.0.0")

model: Any = None
model_load_error: str | None = None
model_path = os.getenv("MODEL_PATH", "/app/model/model.pkl")


class ScoreRequest(BaseModel):
    instances: list[Any] = Field(..., min_length=1)


def _load_model() -> None:
    global model
    global model_load_error

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_load_error = None
    except Exception as exc:  # pragma: no cover
        model = None
        model_load_error = str(exc)


def _to_model_input(instances: list[Any]) -> Any:
    first = instances[0]

    if isinstance(first, dict):
        return pd.DataFrame(instances)

    if isinstance(first, list):
        return np.array(instances)

    return np.array(instances).reshape(-1, 1)


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path,
        "model_load_error": model_load_error,
    }


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded from {model_path}. Error: {model_load_error}",
        )

    if not hasattr(model, "predict"):
        raise HTTPException(status_code=500, detail="Loaded object has no predict() method.")

    try:
        model_input = _to_model_input(request.instances)
        preds = model.predict(model_input)

        if isinstance(preds, np.ndarray):
            predictions = preds.tolist()
        else:
            predictions = list(preds)

        return {"predictions": predictions}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc
