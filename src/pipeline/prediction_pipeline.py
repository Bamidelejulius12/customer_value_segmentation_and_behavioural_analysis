import sys
import os
import pandas as pd
import numpy as np

from src.data.data_processing import preprocess_rfm
from src.modelling.segment_profiling import apply_segment_profile
from src.logger import configure_logger
from src.exception import MyException
from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_cleaning import DataCleaning
from src.features.feature_engineering import FeatureEngineering
from src.utility.model_loader import load_registered_model
from sklearn.preprocessing import StandardScaler
logging = configure_logger()

class Inference:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, feature_engineered_data: pd.DataFrame) -> pd.DataFrame:
        try:
            modelling_data, X_scaled, _ = preprocess_rfm(feature_engineered_data)
            modelling_data = modelling_data.copy()

            modelling_data["Cluster"] = self.model.predict(X_scaled)
            logging.info("customer clustered predicted succesfully")

            modelling_data = apply_segment_profile(modelling_data)
            logging.info("customer segment assigned successfully")

            return modelling_data
        except Exception as e:
            logging.error(f"error occurred while runing prediction {e}")
            raise MyException(e, sys)

def inference_pipeline(model, scaler, feature_engineered_data):
    try:
        inference = Inference(model=model, scaler=scaler)
        predictions = inference.predict(feature_engineered_data)
        logging.info("inference pipeline completed successfully")

        return predictions
    except Exception as e:
        logging.error(f"error occured while running inference pipeline {e}")
        raise MyException(e, sys)



# customer_transaction_data = DataIngestion()
# customer_transaction_data = DataValidation(customer_transaction_data)
# cleaned_customer_transaction_data = DataCleaning(customer_transaction_data)
# customer_transaction_data_profile = FeatureEngineering(cleaned_customer_transaction_data)
# model = load_registered_model()
# scaler = StandardScaler()
# predictions = inference_pipeline(model=model, scaler=scaler, feature_engineered_data=customer_transaction_data_profile)
# print(predictions)