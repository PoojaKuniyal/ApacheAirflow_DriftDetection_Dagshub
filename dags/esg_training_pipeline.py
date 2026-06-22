from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.hooks.base import BaseHook

from src.data_processing import DataProcessing
from src.training import ModelTraining

from config.paths_config import MODEL_OUTPUT_PATH

def preprocess_data():
    conn = BaseHook.get_connection("postgres_default")

    db_params = {
        "host": conn.host,
        "port": conn.port,
        "user": conn.login,
        "password": conn.password,
        "dbname": conn.schema
    }
    processor = DataProcessing(db_params)
    processor.run()

def train_model():
    trainer = ModelTraining(MODEL_OUTPUT_PATH)
    trainer.run()

with DAG(
    dag_id ='esg_training_pipeline',
    start_date = datetime(2025,1,1),
    schedule = None,
    catchup = False
) as dag:
    
    preprocess_task = PythonOperator(
        task_id = 'preprocess_data',
        python_callable = preprocess_data
    )

    train_task = PythonOperator(
        task_id = "train_model",
        python_callable = train_model
    )

    preprocess_task >> train_task