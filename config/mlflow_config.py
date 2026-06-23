import os
import mlflow


def configure_mlflow():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI")
    )