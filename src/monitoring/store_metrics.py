"""Insert data drift and model drift metrics into PostgreSQL."""

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Float, Boolean, DateTime, Integer, String
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

from config.database_config import DB_CONFIG
from config.paths_config import CURRENT_DATA_PATH, CURRENT_PROCESSED_DATA, MODEL_OUTPUT_PATH, X_TRAIN_PATH
from src.monitoring.data_drift import DataDriftMonitor
from src.monitoring.model_drift import ModelDriftMonitor

import mlflow
import mlflow.sklearn
from src.monitoring.mlflow_logger import log_metrics_to_mlflow
from config.mlflow_config import configure_mlflow

configure_mlflow()

Base = declarative_base()


class MonitoringMetrics(Base):
    __tablename__ = "monitoring_metrics"    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    drift_score = Column(Float)
    drift_detected = Column(Boolean)
    drifted_features = Column(String)   # store as comma-separated string or JSON
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)

    accuracy_drop = Column(Float)
    precision_drop = Column(Float)
    recall_drop = Column(Float)
    f1_drop = Column(Float)


class MetricsStore:
    def __init__(self, conn_string):
        self.engine = create_engine(conn_string)
        try:
            Base.metadata.create_all(self.engine) # creates table if not exists
        except OperationalError as exc:
            raise RuntimeError(
                "Could not connect to PostgreSQL. Start the database first "
                "for local runs, for example `astro dev start`, or set "
                "`DATABASE_URL`/Postgres environment variables to the correct host."
            ) from exc
        self.Session = sessionmaker(bind=self.engine)

    def save_metrics(
        self,
        drift_score,
        drift_detected,
        drifted_features,
        accuracy,
        precision,
        recall,
        f1,
        accuracy_drop,
        precision_drop,
        recall_drop,
        f1_drop
    ):
        session = self.Session()
       
        try:
            
            metrics = MonitoringMetrics(
                drift_score=drift_score,
                drift_detected=drift_detected,
                drifted_features=self._format_drifted_features(drifted_features),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,

                accuracy_drop=accuracy_drop, 
                precision_drop= precision_drop,
                recall_drop= recall_drop,
                f1_drop= f1_drop
            )

            session.add(metrics)
            session.commit()
            return metrics.id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_metric_dicts(self, data_drift_metrics, model_metrics):
        """Prepare metrics returned by DataDriftMonitor and ModelDriftMonitor."""
        return {
            "drift_score": data_drift_metrics["drift_score"],
            "drift_detected": data_drift_metrics["drift_detected"],
            "drifted_features": data_drift_metrics["drifted_features"],
            "accuracy": model_metrics["accuracy"],
            "precision": model_metrics["precision"],
            "recall": model_metrics["recall"],
            "f1": model_metrics["f1"],
            "accuracy_drop": model_metrics["accuracy_drop"],
            "precision_drop": model_metrics["precision_drop"],
            "recall_drop": model_metrics["recall_drop"],
            "f1_drop": model_metrics["f1_drop"]
        }
    

    @staticmethod
    def _format_drifted_features(drifted_features):
        if isinstance(drifted_features, list):
            return ",".join(drifted_features)
        return str(drifted_features)

####  If not using airflow we need below code - i.e local machine 
# def build_connection_string(db_config):
#     database_url = os.getenv("DATABASE_URL")
#     if database_url:
#         return database_url

#     host = os.getenv("POSTGRES_HOST", db_config["host"])
#     port = os.getenv("POSTGRES_PORT", db_config["port"])
#     user = os.getenv("POSTGRES_USER", db_config["user"])
#     password = os.getenv("POSTGRES_PASSWORD", db_config["password"])
#     dbname = os.getenv("POSTGRES_DB", db_config["dbname"])

#     return (
#         f"postgresql+psycopg2://{user}:{password}"
#         f"@{host}:{port}/{dbname}"
#     )

# If using airflow

from airflow.hooks.base import BaseHook

def build_connection_string():

    conn = BaseHook.get_connection("postgres_default")

    return (
        f"postgresql+psycopg2://"
        f"{conn.login}:{conn.password}"
        f"@{conn.host}:{conn.port}/{conn.schema}"
    )


def collect_monitoring_metrics():
    data_monitor = DataDriftMonitor()
    data_monitor.load_reference_data(X_TRAIN_PATH)
    data_monitor.load_current_data(CURRENT_DATA_PATH)
    data_drift_metrics = data_monitor.detect_get_drift_summary()

    model_monitor = ModelDriftMonitor(
        MODEL_OUTPUT_PATH,
        CURRENT_PROCESSED_DATA,
        CURRENT_DATA_PATH
    )
    model_metrics = model_monitor.evaluate_current_data()

    return data_drift_metrics, model_metrics

# without airflow conn_string = conn_string or build_connection_string(DB_CONFIG)

def store_monitoring_metrics(conn_string=None):
    conn_string = conn_string or build_connection_string()

    # Compute metrics
    data_drift_metrics, model_metrics = collect_monitoring_metrics()

    # Store metrics in PostgreSQL
    store = MetricsStore(conn_string)
    metric_id = store.save_metric_dicts(
        data_drift_metrics,
        model_metrics
    )
     # Log metrics to DagsHub / MLflow
    log_metrics_to_mlflow(
        data_drift_metrics,
        model_metrics
    )
    return metric_id


if __name__ == "__main__":
    metric_id = store_monitoring_metrics()
    print(f"Saved monitoring metrics with id={metric_id}")
