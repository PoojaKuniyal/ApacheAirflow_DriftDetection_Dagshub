import os

RAW_FILE_PATH = "artifacts/esg_greenwashing_energy_utilities_industrials_2010_2024.csv"
EXTRACT_CSV = "/tmp/raw_data.csv"
TRANSFORM_CSV = "/tmp/transformed_data.csv" 
# /tmp is writable inside containers. if writing to a folder without /tmp then Airflow inside Docker cannot write

# Issue faced - docker container didn't have permission to write into artifacts/ directory


BASE_DIR = os.getcwd()

PROCESSED = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS = os.path.join(BASE_DIR, "artifacts", "models")

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

POWER_TRANSFORMER_PATH = os.path.join(PROCESSED, "powertransformer.pkl")
ENCODED_PATH = os.path.join(PROCESSED, "encoder.pkl")

X_TRAIN_PATH = os.path.join(PROCESSED, "X_train_res.pkl")
Y_TRAIN_PATH = os.path.join(PROCESSED, "y_train_res.pkl")
X_TEST_PATH = os.path.join(PROCESSED, "X_test.pkl")
Y_TEST_PATH = os.path.join(PROCESSED, "y_test.pkl")

MODEL_OUTPUT_PATH = os.path.join(MODELS, "random_forest.pkl")