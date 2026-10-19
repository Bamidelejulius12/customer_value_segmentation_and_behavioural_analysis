import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from src.logger import configure_logger
from src.exception import MyException
import sys
from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_cleaning import DataCleaning
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.features.feature_engineering import FeatureEngineering
from src.data.data_processing import preprocess_rfm
logging = configure_logger()

class Modelling:
    def __init__(self, modelling_data: pd.DataFrame, X_scaled: np.ndarray):
        self.modelling_data = modelling_data.copy()
        self.X_scaled = X_scaled
        self.RANDOM_STATE = 42
        self.sample_size = 5000
        self.k_range = range(2, 11)

    def compute_inertias(self):
        try:
            self.inertias = []

            for k in self.k_range:
                km = KMeans(n_clusters=k, random_state=self.RANDOM_STATE, n_init=10)
                km.fit(self.X_scaled)
                self.inertias.append(km.inertia_)
            return self.inertias
        except Exception as e:
            logging.error(f"error occured while computing inertias {e}")
            raise MyException(e, sys)

    def compute_silhoutte_score(self):
        try:
            self.sil_scores = []

            for k in self.k_range:
                km = KMeans(n_clusters=k, random_state=self.RANDOM_STATE, n_init=10)
                labels = km.fit_predict(self.X_scaled)
                score = silhouette_score(
                    self.X_scaled, 
                    labels, 
                    sample_size=5000, 
                    random_state=self.RANDOM_STATE
                )
                self.sil_scores.append(score)
            return self.sil_scores
        except Exception as e:
            logging.error(f"error occurred when getting sillhoute score {e}")
            raise MyException(e, sys)

    def select_optimal_cluster(self) -> int:
        try:
            if not self.sil_scores:
                raise ValueError("silhoutte score not computed....")
            best_idx = int(np.argmax(self.sil_scores))

            self.chosen_k = list(self.k_range)[best_idx]

            logging.info(f"Best cluster selected {self.chosen_k} "
                         f"silhoutte_score = {self.sil_scores[best_idx]:.4f}")
            return self.chosen_k
        except Exception as e:
            logging.error(f"error occureed while finding the optimal cluster {e}")
            raise MyException(e, sys)

    def fit_kmeans_model(self):
        try:

            self.km_final = KMeans(
                n_clusters=self.chosen_k,
                random_state=self.RANDOM_STATE,
                n_init=10
            )
            self.modelling_data["Cluster"] = self.km_final.fit_predict(self.X_scaled)
            self.final_silhoutte = silhouette_score(
                self.X_scaled,
                self.modelling_data["Cluster"], 
                sample_size=self.sample_size,
                random_state=self.RANDOM_STATE
            )

            logging.info(f"final model cluster = {self.chosen_k}, silhoutte = {self.final_silhoutte}")
            return self.modelling_data["Cluster"]
        except Exception as e:
            logging.error(f"error occurred while fitting the Kmeans model {e}")
            raise MyException(e, sys)

    def run(self):
        self.compute_inertias()
        self.compute_silhoutte_score()
        self.select_optimal_cluster()
        self.fit_kmeans_model()

        return self.km_final, self.chosen_k, self.final_silhoutte


