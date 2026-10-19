import pandas as pd
import numpy as np
from src.logger import configure_logger
from src.exception import MyException
import sys

logging = configure_logger()

def build_cluster_profile(modelling_data: pd.DataFrame) -> pd.DataFrame:
    try:
        profile = (modelling_data
           .groupby("Cluster")
           .agg(Customers=("CustomerID", "count"),
                Recency=("Recency", "mean"),
                Frequency=("Frequency", "mean"),
                Monetary=("Monetary", "mean"),
                AvgR=("R_Score", "mean"),
                AvgF=("F_Score", "mean"),
                AvgM=("M_Score", "mean"))
           .round(2)
           .sort_values("Monetary", ascending=False)
           .reset_index())

        profile["CustomerShare%"] = (100 * profile["Customers"] / profile["Customers"].sum()).round(2)
        profile["RevenueShare%"]  = (100 * (profile["Monetary"] * profile["Customers"])
                                    / (profile["Monetary"] * profile["Customers"]).sum()).round(2)

        logging.info(profile.to_string())

        return profile
    except Exception as e:
        logging.error(f"error occurred while building cluster profile {e}")
        raise MyException(e, sys)

def assign_cluster_names(profile: pd.DataFrame)-> pd.DataFrame:
    try:
        def name_cluster(row):
            
            r = row["Recency"]
            f = row["Frequency"]
            m = row["Monetary"]

            if r <= 15 and f >= 300 and m >= 1_000_000:
                return "Champions"
            elif m >= 1_000_000 and r > 15:
                return "High-Value At Risk"
            elif r <= 10 and f >= 150:
                return "Active Loyal"
            elif r <= 30 and f >= 150:
                return "Loyal Customers"
            elif r > 180 and f < 20:
                return "Dormant / Lost"
            elif r > 60 and f >= 100:
                return "At Risk"
            elif r <= 30 and f < 100:
                return "Emerging Customers"
            elif f < 20 and m < 5_000:
                return "Low-Value Customers"
            elif r <= 180:
                return "Occasional Customers"
            else:
                return "Dormant / Lost"

        profile["SegmentName"] = profile.apply(name_cluster, axis=1)

        return profile

    except Exception as e:
        logging.error(f"error occurred while asssign segment names {e}")
        raise MyException(e, sys)

def apply_segment_profile(modelling_data: pd.DataFrame) -> pd.DataFrame:
    try:
        profile = build_cluster_profile(modelling_data)
        profile = assign_cluster_names(profile)
        
        cluster_to_name = dict(zip(profile["Cluster"], profile["SegmentName"]))
        modelling_data["Segment"] = modelling_data["Cluster"].map(cluster_to_name)
        logging.info("cluster profile and segment names created succesfully")

        return modelling_data
    except Exception as e:
        logging.error(f"error occurred while applying segment profile {e}")
        raise MyException(e, sys)