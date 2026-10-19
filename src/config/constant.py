import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(__file__))))

DB_PATH = os.path.join(BASE_DIR, "customer_data", "vault_chain_flow.db")
TABLE = "Customer_Transactions"

asset_to_currency = {
    "BTC": "BTC",
    "ETH": "ETH",
    "USDT": "USDT",
    "Stablecoin": "USDT",
}

currency_to_asset = {
    "GBP": "Traditional",
    "EUR": "Traditional",
    "USD": "Traditional",
    "USDT": "Stablecoin",
    "BTC": "BTC",
    "ETH": "ETH",
}

FX_TO_GBP = {
    "GBP": 1.0, 
             "EUR": 0.85, 
             "USD": 0.78, 
             "USDT": 0.78,
             "BTC": 42_000.0, 
             "ETH": 2_300.0
             }