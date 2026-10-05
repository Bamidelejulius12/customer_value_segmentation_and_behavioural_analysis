import sqlite3
import sys
import pandas as pd
import numpy as np
from src.config.constant import DB_PATH, TABLE
from src.logger import configure_logger
from src.exception import MyException
from src.data.data_ingestion import DataIngestion

logging = configure_logger()

def DataValidation(customer_transaction_data):
    try:
        col_info = pd.DataFrame({
            "Column":customer_transaction_data.columns,
            "Dtype":customer_transaction_data.dtypes.astype(str).values,
            "Non-Null":customer_transaction_data.notna().sum().values,
            "Null":customer_transaction_data.isna().sum().values,
            "Null %": (customer_transaction_data.isna().mean() * 100).round(2).values,
            "Unique":customer_transaction_data.nunique(dropna=True).values,
        })
        logging.info(col_info.to_string(index=False))
        numeric_cols = customer_transaction_data.select_dtypes(include=[np.number]).columns.tolist()
        logging.info("Numeric columns: %s", numeric_cols)
        logging.info(customer_transaction_data[numeric_cols].describe().T)
        missing = pd.DataFrame({
            "Missing Count": customer_transaction_data.isna().sum(),
            "Missing %": (customer_transaction_data.isna().mean() * 100).round(3),
        }).sort_values("Missing Count", ascending=False)

        logging.info(f"Total missing values in dataset: {customer_transaction_data.isna().sum().sum()}")
        logging.info(missing)
        duplicate_rows = customer_transaction_data.duplicated().sum()
        duplicate_txnids = customer_transaction_data["TransactionID"].duplicated().sum()
        duplicate_custids = customer_transaction_data["CustomerID"].duplicated().sum()

        logging.info(f"Fully duplicated rows: {duplicate_rows}")
        logging.info(f"Duplicated TransactionIDs: {duplicate_txnids}")
        logging.info(f"Duplicated CustomerIDs (expected — one customer, many txns): {duplicate_custids:,}")
        customer_transaction_data["TransactionDate"] = pd.to_datetime(customer_transaction_data["TransactionDate"])
        customer_transaction_data["CustomerDOB"] = pd.to_datetime(customer_transaction_data["CustomerDOB"])

        logging.info(f"Earliest TransactionDate: {customer_transaction_data['TransactionDate'].min()}")
        logging.info(f"Latest   TransactionDate: {customer_transaction_data['TransactionDate'].max()}")
        logging.info(f"Date span (days): {(customer_transaction_data['TransactionDate'].max() - customer_transaction_data['TransactionDate'].min()).days}")

        logging.info(f"\nEarliest CustomerDOB: {customer_transaction_data['CustomerDOB'].min()}")
        logging.info(f"Latest   CustomerDOB: {customer_transaction_data['CustomerDOB'].max()}")

        logging.info(f"\nTransactionTime sample: {customer_transaction_data['TransactionTime'].head(3).tolist()}")

        return customer_transaction_data
    except Exception as e:
        logging.error(f"Error occurred during data validation: {e}")
        raise MyException(e, sys)
