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

from src.config.constant import FX_TO_GBP
logging = configure_logger()

def FeatureEngineering(cleaned_customer_transaction):
    try:
        cleaned_customer_transaction["TransactionDate"] = pd.to_datetime(cleaned_customer_transaction["TransactionDate"])
        cleaned_customer_transaction["CustomerDOB"]     = pd.to_datetime(cleaned_customer_transaction["CustomerDOB"])
        cleaned_customer_transaction["TransactionTime"] = pd.to_datetime(cleaned_customer_transaction["TransactionTime"], format="%H:%M:%S").dt.time

        SNAPSHOT = cleaned_customer_transaction["TransactionDate"].max()
        logging.info(f"Snapshot date:, {SNAPSHOT.date()}")

        cleaned_customer_transaction["AmountGBP"] = cleaned_customer_transaction["TransactionAmount"] * cleaned_customer_transaction["TransactionCurrency"].map(FX_TO_GBP)
        # Revenue-recognised amount: only Completed transactions count toward revenue.
        # Failed / Rejected / Pending / Under Review rows carry 0 revenue.
        cleaned_customer_transaction["RevenueGBP"] = np.where(
            cleaned_customer_transaction["TransactionStatus"] == "Completed",
            cleaned_customer_transaction["AmountGBP"],
            0.0,
        )
        # Time components
        cleaned_customer_transaction["Year"]    = cleaned_customer_transaction["TransactionDate"].dt.year
        cleaned_customer_transaction["Month"]    = cleaned_customer_transaction["TransactionDate"].dt.to_period("M").astype(str)
        cleaned_customer_transaction["DayOfWeek"] = cleaned_customer_transaction["TransactionDate"].dt.day_name()
        cleaned_customer_transaction["Hour"]     = cleaned_customer_transaction["TransactionTime"].apply(lambda t: t.hour)

        # Customer age at time of snapshot
        cleaned_customer_transaction["Age"] = ((SNAPSHOT - cleaned_customer_transaction["CustomerDOB"]).dt.days / 365.25).round(1)

        rfm = cleaned_customer_transaction.groupby("CustomerID").agg(
            Recency=("TransactionDate", lambda s: (SNAPSHOT - s.max()).days),
            Frequency=("TransactionID", "count"),
            Monetary=("RevenueGBP", "sum"),
        ).reset_index()

        logging.info(rfm.head())
        rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

        customer_profile = cleaned_customer_transaction.groupby("CustomerID").agg(
            # Static demographics — take the first value (they don't change)
            CustomerDOB=("CustomerDOB", "first"),
            CustGender=("CustGender", "first"),
            CustLocation=("CustLocation", "first"),
            Age=("Age", "first"),

            # Balance — take the most recent value (it changes over time)
            LastAccountBalance=("CustAccountBalance", "last"),
            FirstAccountBalance=("CustAccountBalance", "first"),
            AvgAccountBalance=("CustAccountBalance", "mean"),

            # Account age / tenure
            FirstTransaction=("TransactionDate", "min"),
            LastTransaction=("TransactionDate", "max"),
        ).reset_index()

        # Derived tenure
        SNAPSHOT = cleaned_customer_transaction["TransactionDate"].max()
        customer_profile["TenureDays"] = (SNAPSHOT - customer_profile["FirstTransaction"]).dt.days
        logging.info(customer_profile.head())

        customer_profile_data = rfm.merge(customer_profile, on="CustomerID", how="left")
        logging.info(customer_profile_data.head())

        return customer_profile_data

    except Exception as e:
        logging.error(f"error occured suring feature engineering.. {e}")
        raise MyException(e, sys)








