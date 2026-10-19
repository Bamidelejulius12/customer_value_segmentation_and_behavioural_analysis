import sys
import os
import mlflow.sklearn

from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

class ModelTracking:
    def __init__(self):
        self.run_name = "customer_segmentation"
        self.registered_model_name = "Vault_Chain_Flow_Segmentation_Model"

    def log_params(self, modelling):
        try:
            mlflow.log_param("chosen_k", int(modelling.chosen_k))
            mlflow.log_param("random_state", int(modelling.RANDOM_STATE))
            mlflow.log_param("k_range_min", int(min(modelling.k_range)))
            mlflow.log_param("k_range_max", int(max(modelling.k_range)))

            logging.info("model parameters has been logged to mlflow")
        except Exception as e:
            logging.error(f"error occurred while logging the mlflow paramters {e}")
            raise MyException(e, sys)

    def log_metrics(self, modelling):
        try:
            mlflow.log_metric("best_sillhoutte_score", float(max(modelling.sil_scores)))
            mlflow.log_metric("final_inertia", float(modelling.km_final.inertia_))

            logging.info("model metrics has been successfully logged")
        except Exception as e:
            logging.error(f"error occured while logging model metrics {e}")
            raise MyException(e, sys)

    def log_model(self, modelling):
        try:
            mlflow.sklearn.log_model(
                sk_model = modelling.km_final,
                artifact_path = "model",
                registered_model_name = self.registered_model_name
            )
            logging.info("model has been successfully logged to mlflow ")
        except Exception as e:
            logging.error(f"error occurred while logging the model to mlflow {e}")
            raise MyException(e, sys)
