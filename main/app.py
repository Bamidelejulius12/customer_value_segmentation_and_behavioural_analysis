import sys
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.visualization.segmentVisualization import segment_overview, segment_behaviour_demographics
from src.visualization.customerVisualization import customer_value_geography
from src.visualization.transactionVisualization import channel_activity, currency_status
from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_cleaning import DataCleaning
from src.pipeline.prediction_pipeline import inference_pipeline
from src.pipeline.training_pipeline import training_pipeline
from src.utility.model_loader import load_registered_model
from src.logger import configure_logger
from src.exception import MyException
from src.features.feature_engineering import FeatureEngineering
from sklearn.preprocessing import StandardScaler
logging = configure_logger()


app = FastAPI(
    title="Vault Chain Flow Analytics API",
    description="API for customer segmentation and transaction analytics visualizations",
    version="1.0.0",
)

MODEL = None
SCALER = None
PREDICTION_DATA = None
TRANSACTION_DATA = None
TRANSACTION_PREDICTION_DATA = None




def figure_to_json_response(fig):
    """
    Serialize a Plotly figure to a JSONResponse safely.

    Plotly figures contain numpy arrays and other non-JSON-serializable
    objects. Using fig.to_json() applies Plotly's encoder (which converts
    numpy types to Python lists), and json.loads() converts the resulting
    JSON string back into a plain dict that JSONResponse can handle.
    """
    return JSONResponse(content=json.loads(fig.to_json()))


def prepare_transaction_data(cleaned_data):
    try:
        cleaned_data = cleaned_data.copy()
        cleaned_data["TransactionDate"] = pd.to_datetime(cleaned_data["TransactionDate"])
        cleaned_data["TransactionTime"] = pd.to_datetime(
            cleaned_data["TransactionTime"].astype(str),
            format="%H:%M:%S",
            errors="coerce",
        ).dt.time
        cleaned_data["Hour"] = cleaned_data["TransactionTime"].apply(
            lambda t: t.hour if pd.notna(t) else None
        )
        logging.info("Transaction data prepared successfully")
        return cleaned_data
    except Exception as e:
        logging.error(f"Error occurred while preparing transaction data: {e}")
        raise MyException(e, sys)


def load_application_data():
    global MODEL
    global SCALER
    global PREDICTION_DATA
    global TRANSACTION_DATA
    global TRANSACTION_PREDICTION_DATA
    try:
        # 1. Load model + scaler from MLflow (once per startup) 
        logging.info("Loading registered model and scaler from MLflow...")
        MODEL = load_registered_model()
        SCALER = StandardScaler()
        logging.info(
            f"Model loaded: {type(MODEL).__name__} "
            f"with n_clusters={getattr(MODEL, 'n_clusters', 'unknown')}"
        )
       

        # 2. Data pipeline 
        logging.info("Ingesting and cleaning data once...")
        customer_data = DataIngestion()
        validated_data = DataValidation(customer_data)
        cleaned_data = DataCleaning(validated_data)
        feature_engineered_data = FeatureEngineering(cleaned_data)
        logging.info("Data ingested and cleaned successfully")

        # 3. Inference (transform-only + predict + segment naming) 
        logging.info("Running inference pipeline...")
        PREDICTION_DATA = inference_pipeline(
            model=MODEL,
            scaler=SCALER,
            feature_engineered_data=feature_engineered_data,
        )
        logging.info(f"Prediction data loaded successfully: {PREDICTION_DATA.shape}")

        # 4. Diagnostics ---
        try:
            n_clusters = PREDICTION_DATA["Cluster"].nunique()
            n_segments = PREDICTION_DATA["Segment"].nunique()
            logging.info(f"Distinct clusters produced: {n_clusters}")
            logging.info(f"Distinct segments produced: {n_segments}")
            logging.info(
                "Segment counts:\n%s",
                PREDICTION_DATA["Segment"].value_counts().to_string(),
            )
        except Exception as diag_err:
            logging.warning(f"Diagnostics step failed (non-fatal): {diag_err}")

        # 5. Transaction-level data -
        logging.info("Preparing transaction-level data...")
        TRANSACTION_DATA = prepare_transaction_data(cleaned_data)
        logging.info("Transaction data prepared successfully")

        TRANSACTION_PREDICTION_DATA = TRANSACTION_DATA.merge(
            PREDICTION_DATA[["CustomerID", "Cluster", "Segment"]],
            on="CustomerID",
            how="left",
        )
        logging.info("Transaction data successfully combined with prediction data")
        logging.info("Application data loaded successfully")

    except Exception as e:
        logging.error(f"Error occurred while loading application data: {e}")
        raise MyException(e, sys)


@app.on_event("startup")
def startup_event():
    try:
        logging.info("Starting Vault Chain Flow Analytics API...")
        load_application_data()
        logging.info("Vault Chain Flow Analytics API startup completed successfully")
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        raise


@app.get("/")
def root():
    return {"message": "Vault Chain Flow Analytics API is running", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "data_loaded": PREDICTION_DATA is not None,
    }


@app.post("/retrain")
def retrain_model():
    try:
        # training_pipeline() logs a new model + scaler to MLflow and registers
        # a new version of Vault_Chain_Flow_Segmentation.
        training_pipeline()
        logging.info("Model retraining completed successfully")

        # Reload the freshly registered model + scaler and re-run inference.
        load_application_data()
        logging.info("Model, predictions and application data refreshed successfully")

        return {
            "status": "success",
            "message": "Model retraining completed successfully",
        }
    except Exception as e:
        logging.error(f"Error occurred during model retraining: {e}")
        raise HTTPException(status_code=500, detail="Model retraining failed")


@app.get("/visualizations/segment-overview")
def get_segment_overview():
    try:
        fig = segment_overview(PREDICTION_DATA)
        return figure_to_json_response(fig)
    except Exception as e:
        logging.error(f"Error generating segment overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate segment overview")


@app.get("/visualizations/segment-behaviour")
def get_segment_behaviour():
    try:
        fig = segment_behaviour_demographics(PREDICTION_DATA)
        return figure_to_json_response(fig)
    except Exception as e:
        logging.error(f"Error generating segment behaviour visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate segment behaviour visualization")


@app.get("/visualizations/customer-value")
def get_customer_value():
    try:
        fig = customer_value_geography(PREDICTION_DATA)
        return figure_to_json_response(fig)
    except Exception as e:
        logging.error(f"Error generating customer value visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate customer value visualization")


@app.get("/visualizations/channel-activity")
def get_channel_activity():
    try:
        type_mix = (
            TRANSACTION_PREDICTION_DATA.groupby(["Segment", "TransactionType"])
            .size()
            .reset_index(name="Count")
        )
        wallet_seg = (
            TRANSACTION_PREDICTION_DATA.groupby("Segment")["WalletActivity"]
            .mean()
            .mul(100)
            .reset_index(name="WalletShare%")
        )
        fig = channel_activity(type_mix, wallet_seg)
        return figure_to_json_response(fig)
    except Exception as e:
        logging.error(f"Error generating channel activity visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate channel activity visualization")


@app.get("/visualizations/currency-status")
def get_currency_status():
    try:
        cur_mix = (
            TRANSACTION_PREDICTION_DATA.groupby(["Segment", "TransactionCurrency"])
            .size()
            .reset_index(name="Count")
        )
        status_mix = (
            TRANSACTION_PREDICTION_DATA.groupby(["Segment", "TransactionStatus"])
            .size()
            .reset_index(name="Count")
        )
        status_mix["Pct"] = status_mix.groupby("Segment")["Count"].transform(
            lambda x: x / x.sum() * 100
        )
        fig = currency_status(cur_mix, status_mix)
        return figure_to_json_response(fig)
    except Exception as e:
        logging.error(f"Error generating currency and status visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate currency and status visualization")

