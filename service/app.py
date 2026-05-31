from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
import time
from typing import Any, Optional
from dataclasses import asdict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from metrics import (
    ERROR_COUNTER,
    PREDICT_COUNTER,
    REQUEST_LATENCY,
)
from model_loader import FeaturePayload, ModelBundle, get_expected_features, load_model_bundle


APP_NAME = "stockout-risk-service"
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/current/model.joblib")
ALLOW_FALLBACK_MODEL = os.getenv("ALLOW_FALLBACK_MODEL", "true").lower() == "true"

model_bundle: Optional[ModelBundle] = None


class PredictRequest(BaseModel):
    sku_id: int = Field(..., description="Идентификатор товара")
    warehouse_id: int = Field(..., description="Идентификатор склада")
    current_stock: float = Field(..., ge=0, description="Текущий остаток")
    safety_stock: float = Field(..., ge=0, description="Страховой запас")
    demand_7d: float = Field(..., ge=0, description="Спрос за 7 дней")
    demand_14d: float = Field(..., ge=0, description="Спрос за 14 дней")
    demand_30d: float = Field(..., ge=0, description="Спрос за 30 дней")
    orders_7d: int = Field(..., ge=0, description="Число заказов за 7 дней")
    orders_14d: int = Field(..., ge=0, description="Число заказов за 14 дней")
    orders_30d: int = Field(..., ge=0, description="Число заказов за 30 дней")
    avg_daily_demand_7d: float = Field(..., ge=0, description="Среднесуточный спрос за 7 дней")
    avg_daily_demand_30d: float = Field(..., ge=0, description="Среднесуточный спрос за 30 дней")
    demand_std_30d: float = Field(..., ge=0, description="Стандартное отклонение спроса за 30 дней")
    demand_trend_7d_vs_prev_7d: float = Field(..., description="Отношение спроса за последние 7 дней к предыдущим 7 дням")
    days_since_replenishment: int = Field(..., ge=0, description="Число дней с последнего пополнения")
    last_replenishment_qty: float = Field(..., ge=0, description="Объем последнего пополнения")
    stock_to_weekly_demand_ratio: float = Field(..., ge=0, description="Отношение остатка к недельному спросу")
    stock_to_safety_ratio: float = Field(..., ge=0, description="Отношение остатка к страховому запасу")
    days_of_cover: float = Field(..., ge=0, description="Дни покрытия остатком")
    day_of_week: int = Field(..., ge=0, le=6, description="День недели")
    month: int = Field(..., ge=1, le=12, description="Месяц")
    category_id: int = Field(..., ge=0, description="Категория товара")
    warehouse_region_id: int = Field(..., ge=0, description="Регион склада")
    warehouse_turnover_rate: float = Field(..., ge=0, description="Скорость оборота склада")


class PredictResponse(BaseModel):
    status: str
    score: float
    stockout_risk: int
    threshold: float
    model_version: str
    model_loaded: bool
    prediction_time_utc: str


def _to_feature_payload(request: PredictRequest) -> FeaturePayload:
    return FeaturePayload(**request.model_dump())


def _build_frame(payload: FeaturePayload) -> pd.DataFrame:
    row = asdict(payload)
    return pd.DataFrame([row], columns=get_expected_features())


def _predict_with_fallback(payload: FeaturePayload) -> tuple[float, int, str]:
    base_score = 0.0

    if payload.current_stock <= payload.safety_stock:
        base_score += 0.45
    if payload.days_of_cover < 7:
        base_score += 0.25
    if payload.demand_trend_7d_vs_prev_7d > 1.15:
        base_score += 0.15
    if payload.days_since_replenishment > 10:
        base_score += 0.10
    if payload.stock_to_weekly_demand_ratio < 1.0:
        base_score += 0.15

    score = min(max(base_score, 0.0), 0.99)
    label = int(score >= 0.50)
    return score, label, "rule-based-fallback"


def _predict(payload: FeaturePayload) -> tuple[float, int, str, bool]:
    global model_bundle

    if model_bundle is None or not model_bundle.loaded:
        if not ALLOW_FALLBACK_MODEL:
            raise HTTPException(status_code=503, detail="Модель еще не загружена")
        score, label, version = _predict_with_fallback(payload)
        return score, label, version, False

    frame = _build_frame(payload)
    model = model_bundle.model

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(frame)[0][1])
    elif hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(frame)[0])
        score = 1.0 / (1.0 + pow(2.718281828, -raw_score))
    else:
        score = float(model.predict(frame)[0])

    threshold = float(model_bundle.threshold)
    label = int(score >= threshold)
    return score, label, model_bundle.version, True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    model_bundle = load_model_bundle(MODEL_PATH)
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    global model_bundle

    loaded = bool(model_bundle and model_bundle.loaded)
    version = model_bundle.version if model_bundle else "not_loaded"

    return {
        "status": "ok",
        "service": APP_NAME,
        "app_version": APP_VERSION,
        "model_loaded": loaded,
        "model_version": version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    global model_bundle

    if model_bundle is None:
        return {
            "status": "ok",
            "model_loaded": False,
            "model_version": "not_loaded",
            "expected_features": get_expected_features(),
        }

    return {
        "status": "ok",
        "model_loaded": model_bundle.loaded,
        "model_version": model_bundle.version,
        "trained_at": model_bundle.trained_at,
        "threshold": model_bundle.threshold,
        "expected_features": get_expected_features(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    started_at = time.perf_counter()

    try:
        payload = _to_feature_payload(request)
        score, label, version, loaded = _predict(payload)
        PREDICT_COUNTER.labels(status="success").inc()
    except HTTPException:
        PREDICT_COUNTER.labels(status="error").inc()
        ERROR_COUNTER.labels(endpoint="/predict", error_type="http_exception").inc()
        raise
    except Exception as exc:
        PREDICT_COUNTER.labels(status="error").inc()
        ERROR_COUNTER.labels(endpoint="/predict", error_type=type(exc).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {exc}") from exc
    finally:
        elapsed = time.perf_counter() - started_at
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed)

    threshold = model_bundle.threshold if model_bundle and loaded else 0.50

    return PredictResponse(
        status="ok",
        score=round(score, 6),
        stockout_risk=label,
        threshold=threshold,
        model_version=version,
        model_loaded=loaded,
        prediction_time_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
