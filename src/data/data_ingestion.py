import sqlite3
import sys
import pandas as pd
import numpy as np
from datetime import datetime

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

from src.config.constant import DB_PATH, TABLE
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

def DataIngestion():
    try:
        con = sqlite3.connect(DB_PATH)
        cursor = con.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        logging.info(f"Tables in database: {cursor.fetchall()}")
        customer_transaction_data = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
        logging.info(f"Loaded {len(customer_transaction_data):,} rows and {customer_transaction_data.shape[1]} columns.")
        logging.info(f"the dataset has a total row of {customer_transaction_data.shape[0]}")
        logging.info(f"the dataset has a total column of {customer_transaction_data.shape[1]}")
        logging.info(customer_transaction_data.head(10))

        return customer_transaction_data

    except Exception as e:
        logging.error(f"Error occurred while ingesting data: {e}")
        raise MyException(e, sys)

