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