# airflow dag and operator- core components to define and schedule tasks

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy
from config.paths_config import *
import logging
from src.custom_exception import *

logger = logging.getLogger("airflow.task")

# EXTRACT
def extract():
    try:
        logger.info("Starting extract task")
        df = pd.read_csv(RAW_FILE_PATH)
        logger.info(f"Rows extracted: {len(df)}")
        df.to_csv(EXTRACT_CSV, index=False)
        logger.info("Extract completed successfully")
    except Exception as e:
        logger.exception("Extract failed")
        raise

# TRANSFORM STEP
def transform():
    try:
        logger.info("Transform started")
        df = pd.read_csv(EXTRACT_CSV)

        # remmove duplicates
        df.drop_duplicates(inplace=True)

        # Create flag -- This marks rows where the year-on-year percentage change is missing (which happens for the first year). 
        df['is_first_year'] = df['yoy_scope1_change_pct'].isnull().astype(int)
        # Impute nulls with 0 (or sentinel like -999)
        df['yoy_scope1_change_pct'] = df['yoy_scope1_change_pct'].fillna(-999)

        # drop
        df.drop(columns=['company','ticker'], inplace=True)
        logger.info("Transformation completed")
        df.to_csv(TRANSFORM_CSV, index=False)

    except Exception as e:
        logger.exception("Transform failed")
        raise   

# LOAD 
def load_to_sql():
    try:
        conn = BaseHook.get_connection('postgres_default')
        engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@esg_ff036e-postgres-1:{conn.port}/{conn.schema}")
        df = pd.read_csv(TRANSFORM_CSV)
        df.to_sql(name='ESG', con=engine, if_exists='replace', index=False)
        logger.info('Loaded to postgres sql....')
    except Exception as e:
        logger.exception("loading to postgres failed")
        raise

# Define the DAG
with DAG(
    dag_id='esg_etl',
    start_date=datetime(2025,1,1),
    schedule=None,
    catchup=False
        ) as dag:
    
    extract_task = PythonOperator(
        task_id = 'extract',
        python_callable = extract
    )

    transform_task  = PythonOperator(
        task_id = 'transform',
        python_callable = transform
    )

    load_task = PythonOperator(
        task_id = 'load',
        python_callable = load_to_sql
    )
    extract_task >> transform_task >> load_task


# You don't pass the DataFrame itself between tasks (Passing DataFrames between tasks doesnt work in airflow because Airflow tasks are isolated)
# i.e transform() doesn't receive the dataframe from extract() automatically
#  Instead, each task operates on files (or tables) so saved each step as csv

# Important: Airflow inside Docker cannot write to ordinary host folders.
# Only /tmp is writable inside containers.
#
# If you try writing to a folder outside /tmp, Airflow tasks running in Docker
# will fail due to permission issues.
#
# Workaround: Always write temporary files to /tmp when running inside containers.
