"""
Centralized Artifact Management

This module provides utilities for saving and loading processed artifacts 
(e.g., transformers, encoders, datasets) in a reproducible way.

- Centralized file:
    Responsible for logging processed artifacts to MLflow/DagsHub so they 
    can be tracked, versioned, and reused across downstream DAG tasks.

- Helper file:
    Handles local persistence and retrieval. Artifacts are saved locally 
    and, if MLflow/DagsHub is available, also logged remotely. When loading, 
    the helper first attempts to read from local storage; if absent, it 
    fetches the latest artifact from MLflow/DagsHub.

This design ensures reproducibility, portability, and resilience across 
Airflow/Docker/DagsHub environments.
"""

import os
import shutil
from contextlib import contextmanager

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.logger import get_logger

logger = get_logger(__name__)

PROCESSED_ARTIFACT_PATH = "processed"
PROCESSED_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_PROCESSED_EXPERIMENT_NAME",
    "ESG_Processed_Artifacts",
)
PROCESSED_RUN_ID_ENV = "MLFLOW_PROCESSED_RUN_ID"


def _configure_experiment():
    mlflow.set_experiment(PROCESSED_EXPERIMENT_NAME)


@contextmanager
def processed_artifact_run(run_name="processed-artifacts"):
    """Create a run for processed artifacts unless the caller already has one."""
    if mlflow.active_run() is not None: 
        yield mlflow.active_run() # guarantees that the artifact logging happens inside a valid MLflow run, without worrying whether one was already active.
        return

    _configure_experiment()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("artifact_stage", "processed")
        yield run


def save_joblib_artifact(obj, local_path, artifact_path=PROCESSED_ARTIFACT_PATH):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    joblib.dump(obj, local_path)
    log_artifact(local_path, artifact_path=artifact_path)


def log_artifact(local_path, artifact_path=PROCESSED_ARTIFACT_PATH):
    try:
        with processed_artifact_run():
            mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)
        logger.info(
            "Logged artifact %s to MLflow artifact path %s",
            local_path,
            artifact_path,
        )
    except Exception as exc:
        logger.warning(
            "Could not log artifact %s to MLflow/DagsHub: %s",
            local_path,
            exc,
        )
        raise


def load_joblib_artifact(local_path, artifact_path=PROCESSED_ARTIFACT_PATH):
    if os.path.exists(local_path):
        return joblib.load(local_path)

    downloaded_path = download_artifact(
        artifact_file=os.path.basename(local_path),
        artifact_path=artifact_path,
    )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy2(downloaded_path, local_path)
    logger.info("Restored artifact %s from MLflow/DagsHub", local_path)
    return joblib.load(local_path)


def download_artifact(artifact_file, artifact_path=PROCESSED_ARTIFACT_PATH):
    run_id = os.getenv(PROCESSED_RUN_ID_ENV) or _get_latest_processed_run_id()
    if not run_id:
        raise FileNotFoundError(
            f"No local artifact found and no MLflow run found for {artifact_file}. "
            f"Set {PROCESSED_RUN_ID_ENV} to a run containing processed artifacts."
        )

    client = MlflowClient()
    remote_path = f"{artifact_path}/{artifact_file}"
    return client.download_artifacts(run_id=run_id, path=remote_path)


def _get_latest_processed_run_id():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(PROCESSED_EXPERIMENT_NAME)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=("attributes.status = 'FINISHED'"
                       "and tags.artifact_stage = 'processed'"), # "and tags.artifact_stage = 'processed'" avoids accidentally downloading some unrelated run.
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None

    return runs[0].info.run_id
