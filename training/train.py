from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
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

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "stockout-risk-training"


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

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 4,
        "random_state": 42,
        "class_weight": "balanced",
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random_forest_stockout") as run:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"],
            class_weight=params["class_weight"],
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        score = model.predict_proba(X_valid)[:, 1]
        pred = (score >= 0.50).astype(int)

        metrics = {
            "recall_stockout": recall_score(y_valid, pred),
            "precision_stockout": precision_score(y_valid, pred),
            "roc_auc": roc_auc_score(y_valid, score),
            "pr_auc": average_precision_score(y_valid, score),
        }

        mlflow.log_params(params)
        mlflow.log_metric("validation_rows", len(X_valid))
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="stockout-risk-model",
        )

        artifact = {
            "model": model,
            "version": f"rf-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "threshold": 0.50,
            "features": FEATURES,
            "validation_rows": len(X_valid),
            "mlflow_run_id": run.info.run_id,
        }

        output_path = ARTIFACTS_DIR / "model.joblib"
        joblib.dump(artifact, output_path)

        print(f"Сохранена модель: {output_path}")
        print(f"MLflow run_id: {run.info.run_id}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()