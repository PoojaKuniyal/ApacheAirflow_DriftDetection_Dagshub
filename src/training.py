import joblib
import numpy as np
import os
from src.logger import *
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_parameters import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from src.artifact_store import load_joblib_artifact, save_joblib_artifact

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, model_output_path):
        
        self.model_output_path = model_output_path
        self.param_dist = RANDOM_FOREST_PARAMETERS
        self.random_search_param = RANDOM_SEARCH_PARAM

        logger.info("Model training initailized..")

    def load_data(self):
        try:
            X_train = load_joblib_artifact(X_TRAIN_PATH)
            X_test = load_joblib_artifact(X_TEST_PATH)
            y_train = load_joblib_artifact(Y_TRAIN_PATH)
            y_test = load_joblib_artifact(Y_TEST_PATH)

            logger.info("Data loaded successfully...")
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            raise CustomException("Failed to load data",e)
    
    def train_model(self, X_train,y_train):
        try:
            logger.info('Initializing training...')
    
            random_forest_model = RandomForestClassifier(random_state=42)

            logger.info('starting hyperparameter tuninig...')
            random_search = RandomizedSearchCV(
                estimator= random_forest_model,
                param_distributions= self.param_dist,
                n_iter= self.random_search_param['n_iter'],
                cv = self.random_search_param['cv'],
                n_jobs= self.random_search_param['n_jobs'],
                random_state= self.random_search_param['random_state'],
                scoring= self.random_search_param['scoring']
            )
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_rf_model = random_search.best_estimator_
            logger.info(f"Best paramters are : {best_params}")
            return best_rf_model, best_params   
        
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model" ,  e)

    def evaluate(self, model, X_test, y_test):
        try:
            logger.info('Evaluating the model...')
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")

            return{
                'accuracy' : accuracy,
                'precision' : precision,
                'recall' : recall,
                'f1' : f1
            }
        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model" ,  e)


    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info('saving the model')

            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model" ,  e)
 
    def run(self):
        try:
           
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

            # Set experiment name
            mlflow.set_experiment("RF_Experiment1") # experiment should be set before starting the run

            experiment = mlflow.get_experiment_by_name("RF_Experiment1")
            print(experiment)

            with mlflow.start_run():
                logger.info('Starting our ML training pipeline and ML Flow experimentation')
                
                X_train,X_test,y_train,y_test = self.load_data()
                best_rf_model, best_params = self.train_model(X_train,y_train)
                metrics = self.evaluate(best_rf_model, X_test, y_test)
                self.save_model(best_rf_model)
                # 🔑 Dump metrics dynamically instead of hardcoding
                # Define the directory and file path separately
                BASELINE_DIR = PROCESSED
                BASELINE_FILE = os.path.join(BASELINE_DIR, "baseline.pkl")

                # Make sure the directory exists
                os.makedirs(BASELINE_DIR, exist_ok=True)

                # Dump metrics into the file
                save_joblib_artifact(metrics, BASELINE_FILE)

                logger.info('Logging the model into ML flow')
                mlflow.sklearn.log_model(
                    sk_model=best_rf_model,
                    name="random_forest_model",
                    #serialization_format="skops" 
                )

                logger.info('Logging param and metrics to ML Flow')
                mlflow.log_params(best_params)

                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                mlflow.set_tags({"model_type": "RandomForest", "stage": "development"})

                # Log confusion matrix or feature importance as artifacts
                disp = ConfusionMatrixDisplay.from_estimator(best_rf_model, X_test, y_test)
                plt.savefig("confusion_matrix.png")
                plt.close()
                mlflow.log_artifact(local_path="confusion_matrix.png",
                                   artifact_path="plots")

                importance = pd.DataFrame(best_rf_model.feature_importances_, index=X_train.columns, columns=["importance"])
                importance.to_csv("feature_importance.csv")
                mlflow.log_artifact(local_path ="feature_importance.csv",
                                    artifact_path="reports")

                logger.info('Model training successfully completed...')
    

        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed during model training pipeline" ,  e)

if __name__ == '__main__':
    trainer = ModelTraining(MODEL_OUTPUT_PATH)
    trainer.run()

# Issue: The local `mlflow ui` shows nothing because experiment runs are stored inside the Docker container,
# while the local MLflow UI command points to a different SQLite database.
#
# Solution: Run MLflow in its own dedicated container. Do not rely on Astro's internal MLflow.
# Instead, separate MLflow from Astro entirely and manage it independently.
#
# Note: Set `mlflow.set_tracking_uri("http://host.docker.internal:5000")`
# because Airflow DAGs run inside Docker and must connect to the MLflow server
# running on your Windows host.
