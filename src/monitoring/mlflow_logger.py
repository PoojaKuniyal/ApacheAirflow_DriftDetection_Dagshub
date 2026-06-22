from datetime import datetime
import mlflow

def log_metrics_to_mlflow(data_drift_metrics, model_metrics):

    mlflow.set_experiment("RF_Monitoring")

    with mlflow.start_run(
        run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        mlflow.log_metric(
            "drift_score",
            data_drift_metrics["drift_score"]
        )

        mlflow.log_metric(
            "drift_detected",
            int(data_drift_metrics["drift_detected"])
        )

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "accuracy_drop",
            "precision_drop",
            "recall_drop",
            "f1_drop"
        ]

        for metric in metric_names:
            mlflow.log_metric(metric, model_metrics[metric])

        mlflow.log_param(
            "drifted_features",
            ",".join(data_drift_metrics["drifted_features"])
        )