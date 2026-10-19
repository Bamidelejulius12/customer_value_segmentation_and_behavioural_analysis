import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_rfm(modelling_data):
    X = modelling_data[["Recency", "Frequency", "Monetary"]].copy()

    # Log-transform the skewed features (Recency stays raw)
    X["Recency"]  = np.log1p(X["Recency"])
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])

    # Standardise so all three features contribute equally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return modelling_data, X_scaled, scaler
