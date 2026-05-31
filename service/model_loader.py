from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


EXPECTED_FEATURES = [
    "sku_id",
    "warehouse_id",
    "current_stock",
    "safety_stock",
    "demand_7d",
    "demand_14d",
    "demand_30d",
    "orders_7d",
    "orders_14d",
    "orders_30d",
    "avg_daily_demand_7d",
    "avg_daily_demand_30d",
    "demand_std_30d",
    "demand_trend_7d_vs_prev_7d",
    "days_since_replenishment",
    "last_replenishment_qty",
    "stock_to_weekly_demand_ratio",
    "stock_to_safety_ratio",
    "days_of_cover",
    "day_of_week",
    "month",
    "category_id",
    "warehouse_region_id",
    "warehouse_turnover_rate",
]


@dataclass
class FeaturePayload:
    sku_id: int
    warehouse_id: int
    current_stock: float
    safety_stock: float
    demand_7d: float
    demand_14d: float
    demand_30d: float
    orders_7d: int
    orders_14d: int
    orders_30d: int
    avg_daily_demand_7d: float
    avg_daily_demand_30d: float
    demand_std_30d: float
    demand_trend_7d_vs_prev_7d: float
    days_since_replenishment: int
    last_replenishment_qty: float
    stock_to_weekly_demand_ratio: float
    stock_to_safety_ratio: float
    days_of_cover: float
    day_of_week: int
    month: int
    category_id: int
    warehouse_region_id: int
    warehouse_turnover_rate: float


@dataclass
class ModelBundle:
    model: Any
    version: str
    trained_at: str
    threshold: float
    loaded: bool


def get_expected_features() -> list[str]:
    return EXPECTED_FEATURES.copy()


def load_model_bundle(model_path: str) -> ModelBundle:
    path = Path(model_path)

    if not path.exists():
        return ModelBundle(
            model=None,
            version="not_loaded",
            trained_at="unknown",
            threshold=0.50,
            loaded=False,
        )

    artifact = joblib.load(path)

    if isinstance(artifact, dict):
        return ModelBundle(
            model=artifact.get("model"),
            version=str(artifact.get("version", "unknown")),
            trained_at=str(artifact.get("trained_at", "unknown")),
            threshold=float(artifact.get("threshold", 0.50)),
            loaded=True,
        )

    return ModelBundle(
        model=artifact,
        version="unknown",
        trained_at="unknown",
        threshold=0.50,
        loaded=True,
    )
