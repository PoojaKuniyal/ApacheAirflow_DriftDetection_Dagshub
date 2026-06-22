
import os

RAW_FILE_PATH = "artifacts/esg_greenwashing_energy_utilities_industrials_2010_2024.csv"
EXTRACT_CSV = "/tmp/raw_data.csv"
TRANSFORM_CSV = "/tmp/transformed_data.csv" 

BASE_DIR = os.getcwd()

PROCESSED = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS = os.path.join(BASE_DIR, "artifacts", "models")

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

CURRENT_DATA_PATH = 'artifacts/esg_2024_2026_combined.csv'

CURRENT_PROCESSED_DATA = os.path.join(PROCESSED,'current.pkl')

POWER_TRANSFORMER_PATH = os.path.join(PROCESSED, "powertransformer.pkl")
ENCODED_PATH = os.path.join(PROCESSED, "encoder.pkl")

X_TRAIN_PATH = os.path.join(PROCESSED, "X_train_res.pkl")
Y_TRAIN_PATH = os.path.join(PROCESSED, "y_train_res.pkl")
X_TEST_PATH = os.path.join(PROCESSED, "X_test.pkl")
Y_TEST_PATH = os.path.join(PROCESSED, "y_test.pkl")

MODEL_OUTPUT_PATH = os.path.join(MODELS, "random_forest.pkl")

# ───────────────────────────────────────────────
# Airflow + Docker Notes (ETL Artifacts Handling)
# ───────────────────────────────────────────────

# /tmp is writable inside containers.
#   - If writing to a folder outside /tmp, Airflow inside Docker may not have permission.
#   - Issue faced: docker container couldn't write into artifacts/ directory.

# ⚠️ Important:
#   - Do NOT store processed artifacts in /tmp in production.
#   - /tmp is for temporary files only.
#   - For production, use long-term storage.

# Artifact Store:
#   - Using DagsHub as the artifact store in this project.
#   - Benefit: monitoring DAG becomes independent of training DAG.
#   - If artifacts are stored in /tmp, the monitoring DAG depends on the training DAG having run recently.
#   - With DagsHub, DAGs can run independently.
# DVC (and others artifact store) can also be used. 