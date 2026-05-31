from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "training" / "artifacts"

FEATURES = [
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
TARGET = "stockout_target"


def main() -> None:
    dataset_path = ARTIFACTS_DIR / "training_dataset.csv"
    model_path = ARTIFACTS_DIR / "model.joblib"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Не найден датасет {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Не найдена модель {model_path}")

    df = pd.read_csv(dataset_path)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.50))

    X = df[FEATURES]
    y = df[TARGET]

    score = model.predict_proba(X)[:, 1]
    pred = (score >= threshold).astype(int)

    metrics = {
        "recall_stockout": recall_score(y, pred),
        "precision_stockout": precision_score(y, pred),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
    }

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)

    print("Результаты оценки:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Метрики сохранены в {metrics_path}")


if __name__ == "__main__":
    main()
