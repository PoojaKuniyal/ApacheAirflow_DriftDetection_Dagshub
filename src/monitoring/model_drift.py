import joblib
import os
from config.paths_config import *
import pandas as pd
from src.artifact_store import load_joblib_artifact

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class ModelDriftMonitor:

    def __init__(self, model_path, processed_current_data, current_data):

        self.model = joblib.load(model_path)
        self.processed_current_data = load_joblib_artifact(processed_current_data)
        self.current_data = current_data 
        
    def evaluate_current_data(self):

        X_current = self.processed_current_data
        data = pd.read_csv(self.current_data)
        y_current = data['greenwashing_flag'] 

        y_pred = self.model.predict(X_current)

        BASELINE_METRIC_FILE = os.path.join(PROCESSED, "baseline.pkl")

        try:
            BASELINE_METRICS = load_joblib_artifact(BASELINE_METRIC_FILE)
        except FileNotFoundError:
            # First run: no baseline yet
            BASELINE_METRICS = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
            print("⚠️ No baseline metrics found, starting fresh.")

        # Compute current metrics
        accuracy = round(accuracy_score(y_current, y_pred), 4)
        precision = round(precision_score(y_current, y_pred, zero_division=0), 4)
        recall = round(recall_score(y_current, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_current, y_pred, zero_division=0), 4)

        # Add both metrics and drops
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy_drop": round(BASELINE_METRICS["accuracy"] - accuracy, 4),
            "precision_drop": round(BASELINE_METRICS["precision"] - precision, 4),
            "recall_drop": round(BASELINE_METRICS["recall"] - recall, 4),
            "f1_drop": round(BASELINE_METRICS["f1"] - f1, 4),
        }

        return metrics


if __name__ =='__main__':

    monitor = ModelDriftMonitor(
        MODEL_OUTPUT_PATH, CURRENT_PROCESSED_DATA, CURRENT_DATA_PATH)

    metrics = monitor.evaluate_current_data()

    print(metrics)
