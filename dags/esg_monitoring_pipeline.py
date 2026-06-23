from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.hooks.base import BaseHook
from src.monitoring.store_metrics import store_monitoring_metrics

with DAG(
    dag_id="monitoring_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    monitoring_task = PythonOperator(
        task_id="monitoring", python_callable=store_monitoring_metrics
    )
