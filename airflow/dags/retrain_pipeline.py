from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "ekaterina",
    "depends_on_past": False,
    "retries": 1,
}


with DAG(
    dag_id="retrain_stockout_model",
    default_args=default_args,
    description="Пайплайн подготовки данных, обучения, оценки и регистрации модели риска дефицита",
    start_date=datetime(2026, 5, 31),
    schedule="@daily",
    catchup=False,
    tags=["ml", "stockout", "retraining"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /opt/airflow && python /opt/airflow/training/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        bash_command="cd /opt/airflow && python /opt/airflow/training/train.py",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="cd /opt/airflow && python /opt/airflow/training/evaluate.py",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="cd /opt/airflow && python /opt/airflow/training/register_model.py",
    )

    preprocess >> train >> evaluate >> register_model
