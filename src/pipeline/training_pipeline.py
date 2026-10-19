from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_cleaning import DataCleaning
from src.features.feature_engineering import FeatureEngineering
from src.data.data_processing import preprocess_rfm
from src.modelling.model_training import Modelling
from src.modelling.model_tracking import ModelTracking
from src.logger import configure_logger
from src.exception import MyException
from src.utility.mlflow_setup import setup_mlflow
import sys
import mlflow
logging = configure_logger()

def training_pipeline():
    try:
        setup_mlflow()
        with mlflow.start_run(run_name = "customer_segmenataion") as run:

            customer_transaction_data = DataIngestion()
            customer_transaction_data = DataValidation(customer_transaction_data)
            cleaned_customer_transaction_data = DataCleaning(customer_transaction_data)
            customer_transaction_data_profile = FeatureEngineering(cleaned_customer_transaction_data)
            modelling_data, X_scaled, _ = preprocess_rfm(customer_transaction_data_profile)
            modelling = Modelling(modelling_data=modelling_data, X_scaled = X_scaled)
            modelling.run()

            model_tracking = ModelTracking()
            model_tracking.log_params(modelling)
            model_tracking.log_metrics(modelling)
            model_tracking.log_model(modelling)

            logging.info(f"chosen k {modelling.chosen_k}")
            logging.info(f"sillhoutte score {modelling.final_silhoutte:.4f}")
            logging.info("Mlflow model tracking completed successfully")
    except Exception as e:
        logging.error(f"error occured while running training pipeline {e}")
        raise MyException(e, sys)


# training_pipeline()