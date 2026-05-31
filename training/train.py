from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "training" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Не найден датасет {dataset_path}. Сначала запусти preprocess.py"
        )

    df = pd.read_csv(dataset_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "version": f"rf-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": 0.50,
        "features": FEATURES,
        "validation_rows": len(X_valid),
    }

    output_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(artifact, output_path)
    print(f"Сохранена модель: {output_path}")


if __name__ == "__main__":
    main()
