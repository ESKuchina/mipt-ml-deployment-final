from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "training" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_training_dataset(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    sku_id = rng.integers(1, 500, size=n_rows)
    warehouse_id = rng.integers(1, 20, size=n_rows)
    current_stock = rng.integers(0, 400, size=n_rows).astype(float)
    safety_stock = rng.integers(10, 120, size=n_rows).astype(float)

    demand_7d = rng.uniform(0, 200, size=n_rows)
    demand_14d = demand_7d + rng.uniform(0, 180, size=n_rows)
    demand_30d = demand_14d + rng.uniform(0, 250, size=n_rows)

    orders_7d = rng.integers(0, 50, size=n_rows)
    orders_14d = orders_7d + rng.integers(0, 45, size=n_rows)
    orders_30d = orders_14d + rng.integers(0, 60, size=n_rows)

    avg_daily_demand_7d = demand_7d / 7
    avg_daily_demand_30d = demand_30d / 30
    demand_std_30d = rng.uniform(0, 25, size=n_rows)
    demand_trend_7d_vs_prev_7d = rng.uniform(0.5, 1.8, size=n_rows)

    days_since_replenishment = rng.integers(0, 25, size=n_rows)
    last_replenishment_qty = rng.uniform(0, 300, size=n_rows)

    stock_to_weekly_demand_ratio = np.divide(
        current_stock,
        np.where(demand_7d == 0, 1.0, demand_7d),
    )
    stock_to_safety_ratio = np.divide(
        current_stock,
        np.where(safety_stock == 0, 1.0, safety_stock),
    )
    days_of_cover = np.divide(
        current_stock,
        np.where(avg_daily_demand_7d == 0, 1.0, avg_daily_demand_7d),
    )

    day_of_week = rng.integers(0, 7, size=n_rows)
    month = rng.integers(1, 13, size=n_rows)
    category_id = rng.integers(1, 15, size=n_rows)
    warehouse_region_id = rng.integers(1, 8, size=n_rows)
    warehouse_turnover_rate = rng.uniform(0.1, 3.5, size=n_rows)

    risk_score = (
        1.2 * (current_stock <= safety_stock).astype(float)
        + 1.3 * (days_of_cover < 7).astype(float)
        + 0.8 * (demand_trend_7d_vs_prev_7d > 1.15).astype(float)
        + 0.6 * (days_since_replenishment > 10).astype(float)
        + 1.0 * (stock_to_weekly_demand_ratio < 1.0).astype(float)
        + 0.03 * demand_std_30d
        + rng.normal(0, 0.35, size=n_rows)
    )

    stockout_target = (risk_score > 2.0).astype(int)

    df = pd.DataFrame(
        {
            "sku_id": sku_id,
            "warehouse_id": warehouse_id,
            "current_stock": current_stock,
            "safety_stock": safety_stock,
            "demand_7d": demand_7d,
            "demand_14d": demand_14d,
            "demand_30d": demand_30d,
            "orders_7d": orders_7d,
            "orders_14d": orders_14d,
            "orders_30d": orders_30d,
            "avg_daily_demand_7d": avg_daily_demand_7d,
            "avg_daily_demand_30d": avg_daily_demand_30d,
            "demand_std_30d": demand_std_30d,
            "demand_trend_7d_vs_prev_7d": demand_trend_7d_vs_prev_7d,
            "days_since_replenishment": days_since_replenishment,
            "last_replenishment_qty": last_replenishment_qty,
            "stock_to_weekly_demand_ratio": stock_to_weekly_demand_ratio,
            "stock_to_safety_ratio": stock_to_safety_ratio,
            "days_of_cover": days_of_cover,
            "day_of_week": day_of_week,
            "month": month,
            "category_id": category_id,
            "warehouse_region_id": warehouse_region_id,
            "warehouse_turnover_rate": warehouse_turnover_rate,
            "stockout_target": stockout_target,
        }
    )

    return df


def main() -> None:
    df = build_training_dataset()
    output_path = ARTIFACTS_DIR / "training_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Подготовлен датасет: {output_path}")


if __name__ == "__main__":
    main()
