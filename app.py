import mlflow.xgboost
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date, timedelta

import os

app = FastAPI(title="Weekly Sales Forecasting API")

RUN_ID = "1c6b07c974cc4ac190b824884e5002bc"
model  = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/model")


FEATURES = [
    "year","weekofyear", "month", "quarter", "sales_lag_52"
]

# --- Schemas ---
class SingleRequest(BaseModel):
    date:          date
    lag_52:        float    # sales same week last year


class SingleResponse(BaseModel):
    week_start:      str
    predicted_sales: float


# --- Helpers ---
def next_monday(d: date) -> date:
    """Snap a date to the nearest Monday (week start)."""
    return d - timedelta(days=d.weekday())


def build_row(d: date, promo: int, history: list[float]) -> dict:
    return {
        "year":          d.year,
        "weekofyear":  d.isocalendar()[1],
        "month":         d.month,
        "quarter":       (d.month - 1) // 3 + 1,
        "sales_lag_52":        history[-52],
    }


# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "model": "xgboost-weekly-sales-forecasting"}


@app.post("/predict/single", response_model=SingleResponse)
def predict_single(req: SingleRequest):
    """Predict sales for a single week given pre-computed features."""
    row  = {
        "year":           req.date.year,
        "weekofyear":   req.date.isocalendar()[1],
        "month":          req.date.month,
        "quarter":        (req.date.month - 1) // 3 + 1,
        "sales_lag_52":         req.lag_52,
    }

    X    = pd.DataFrame([row])[FEATURES]
    pred = float(model.predict(X)[0])

    return SingleResponse(
        week_start=str(next_monday(req.date)),
        predicted_sales=round(pred, 2)
    )


