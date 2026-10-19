import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.utility.mlflow_setup import setup_mlflow
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()


def load_registered_model(model_name: str = "Vault_Chain_Flow_Segmentation_Model"):
    """
    Load the latest registered KMeans model from the configured MLflow tracking server.
    """
    try:
        setup_mlflow()
        client = MlflowClient()

        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            raise ValueError(f"No versions found for model '{model_name}'")

        
        latest_model_version = sorted(model_versions, key=lambda v: int(v.version))[-1]
        logging.info(f"Loading model '{model_name}' version {latest_model_version.version}")

        model = mlflow.sklearn.load_model(
            f"models:/{model_name}/{latest_model_version.version}"
        )
        return model

    except Exception as e:
        logging.error(f"Error loading model from MLflow: {e}")
        raise MyException(e, sys)