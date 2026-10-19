
import json
import os

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Vault Chain Flow Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        div.stButton > button {
            width: 100%;
            min-height: 42px;
            font-weight: 600;
            border-radius: 8px;
        }

        div[data-testid="stMetric"] {
            padding: 14px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL.rstrip("/")

if "figures" not in st.session_state:
    st.session_state.figures = {}

if "last_retrain" not in st.session_state:
    st.session_state.last_retrain = None


# --------------------------------------------------
# API HELPERS
# --------------------------------------------------
def api_url(endpoint: str) -> str:
    """Build a URL for the FastAPI service."""
    return f"{st.session_state.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def check_api_health():
    """Check whether the FastAPI service is reachable."""
    try:
        response = requests.get(
            api_url("/health"),
            timeout=10,
        )
        response.raise_for_status()
        return response.json(), None

    except requests.RequestException as exc:
        return None, str(exc)


def fetch_visualization(endpoint: str):
    """Fetch a Plotly figure serialized by the FastAPI endpoint."""
    try:
        response = requests.get(
            api_url(endpoint),
            timeout=60,
        )
        response.raise_for_status()

        figure_data = response.json()

        # FastAPI returns fig.to_json() converted to a dictionary.
        figure = pio.from_json(json.dumps(figure_data))

        return figure, None

    except requests.HTTPError as exc:
        message = (
            f"API returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:500]}"
        )
        return None, message

    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, str(exc)


def display_visualization(
    key: str,
    endpoint: str,
    button_label: str,
    chart_title: str,
    help_text: str = "",
):
    """Load and display a visualization when its button is clicked."""

    if help_text:
        st.caption(help_text)

    if st.button(button_label, key=f"load_{key}"):
        with st.spinner(f"Loading {chart_title.lower()}..."):
            figure, error = fetch_visualization(endpoint)

        if error:
            st.error(f"Could not load {chart_title}: {error}")
        else:
            st.session_state.figures[key] = figure

    figure = st.session_state.figures.get(key)

    if figure is not None:
        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"chart_{key}",
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("Vault Chain Flow")
    st.caption("Customer Segmentation & Transaction Analytics")

    st.divider()

    st.subheader("API Configuration")

    api_input = st.text_input(
        "FastAPI base URL",
        value=st.session_state.api_base_url,
        help="Example: http://127.0.0.1:8000",
    )

    if st.button("Connect to API", key="connect_api"):
        st.session_state.api_base_url = api_input.strip().rstrip("/")
        st.session_state.figures = {}
        st.rerun()

    st.divider()

    if st.button("Check API Health", key="check_health"):
        health, error = check_api_health()

        if error:
            st.error("API connection failed.")
            st.caption(error)
        else:
            st.success("API is reachable.")

            st.write(
                "Model loaded:",
                "Yes" if health.get("model_loaded") else "No",
            )
            st.write(
                "Data loaded:",
                "Yes" if health.get("data_loaded") else "No",
            )

            st.json(health)

    st.divider()

    st.caption("Backend: FastAPI")
    st.caption("Model registry: MLflow")


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------
st.title("Vault Chain Flow Analytics")
st.write(
    "Monitor customer segments, understand customer value, "
    "analyse transaction behaviour, and retrain the segmentation model."
)

health, health_error = check_api_health()

if health_error:
    st.warning(
        "The API is currently unreachable. Start your FastAPI server "
        "and confirm the configured base URL."
    )
else:
    model_status = (
        "Loaded" if health.get("model_loaded") else "Not loaded"
    )
    data_status = (
        "Loaded" if health.get("data_loaded") else "Not loaded"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("API Status", "Online")

    with col2:
        st.metric("Segmentation Model", model_status)

    with col3:
        st.metric("Prediction Data", data_status)


st.divider()


# --------------------------------------------------
# FOUR DASHBOARD TABS
# --------------------------------------------------
(
    retrain_tab,
    segment_tab,
    customer_tab,
    transaction_tab,
) = st.tabs(
    [
        "Model Retraining",
        "Segment Analytics",
        "Customer Analytics",
        "Transaction Analytics",
    ]
)


# ==================================================
# TAB 1: MODEL RETRAINING
# ==================================================
with retrain_tab:
    st.header("Model Retraining")

    st.write(
        "Run the training pipeline to train and register an updated "
        "customer segmentation model. After training, the API reloads "
        "the registered model and refreshes its predictions and "
        "transaction-level data."
    )

    st.warning(
        "Retraining can take some time. Avoid triggering multiple "
        "retraining requests simultaneously."
    )

    st.subheader("Current Model Status")

    if health_error:
        st.error("Unable to retrieve model status because the API is offline.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Model",
                "Available" if health.get("model_loaded") else "Unavailable",
            )

        with col2:
            st.metric(
                "Prediction Data",
                "Available" if health.get("data_loaded") else "Unavailable",
            )

    st.divider()

    st.subheader("Retraining Controls")

    confirm_retrain = st.text_input(
        "Enter RETRAIN to enable the operation",
        placeholder="RETRAIN",
        key="retrain_confirmation",
    )

    retrain_button = st.button(
        "Start Model Retraining",
        type="primary",
        disabled=(
            confirm_retrain.strip() != "RETRAIN"
            or health_error is not None
        ),
        key="start_retraining",
    )

    if retrain_button:
        try:
            with st.spinner(
                "Retraining model, registering the new version, "
                "and refreshing predictions..."
            ):
                response = requests.post(
                    api_url("/retrain"),
                    timeout=(10, 1800),
                )
                response.raise_for_status()

                result = response.json()

            st.session_state.last_retrain = result
            st.session_state.figures = {}

            st.success("Retraining completed successfully.")

            st.json(result)

            st.info(
                "The API has reloaded its model and prediction data. "
                "Load the visualizations again to see the refreshed results."
            )

            st.rerun()

        except requests.HTTPError as exc:
            st.error(
                f"Retraining failed with HTTP "
                f"{exc.response.status_code}."
            )

            try:
                st.json(exc.response.json())
            except ValueError:
                st.code(exc.response.text[:1000])

        except requests.RequestException as exc:
            st.error(f"Could not complete the retraining request: {exc}")

    if st.session_state.last_retrain:
        st.divider()
        st.subheader("Last Successful Retraining Response")
        st.json(st.session_state.last_retrain)


# ==================================================
# TAB 2: SEGMENT ANALYTICS
# ==================================================
with segment_tab:
    st.header("Customer Segment Analytics")

    st.write(
        "Explore the distribution of customer segments and compare "
        "their behavioural patterns and demographic characteristics."
    )

    overview_col, behaviour_col = st.columns(2)

    with overview_col:
        st.subheader("Segment Overview")

        display_visualization(
            key="segment_overview",
            endpoint="/visualizations/segment-overview",
            button_label="Load Segment Overview",
            chart_title="Segment Overview",
            help_text=(
                "View the distribution and relative size of the "
                "identified customer segments."
            ),
        )

    with behaviour_col:
        st.subheader("Segment Behaviour & Demographics")

        display_visualization(
            key="segment_behaviour",
            endpoint="/visualizations/segment-behaviour",
            button_label="Load Segment Behaviour",
            chart_title="Segment Behaviour & Demographics",
            help_text=(
                "Compare the characteristics and behaviour of "
                "customers across segments."
            ),
        )


# ==================================================
# TAB 3: CUSTOMER ANALYTICS
# ==================================================
with customer_tab:
    st.header("Customer Value Analytics")

    st.write(
        "Explore customer value and geographical patterns to understand "
        "where valuable customers are located and how customer value "
        "varies across the available geography."
    )

    display_visualization(
        key="customer_value",
        endpoint="/visualizations/customer-value",
        button_label="Load Customer Value Analysis",
        chart_title="Customer Value & Geography",
        help_text=(
            "The chart is generated by the API using the current "
            "customer segmentation predictions."
        ),
    )


# ==================================================
# TAB 4: TRANSACTION ANALYTICS
# ==================================================
with transaction_tab:
    st.header("Transaction Analytics")

    st.write(
        "Analyse transaction channel activity, wallet usage, currency "
        "distribution, and transaction outcomes across customer segments."
    )

    channel_tab, currency_tab = st.columns(2)

    with channel_tab:
        st.subheader("Channel & Wallet Activity")

        display_visualization(
            key="channel_activity",
            endpoint="/visualizations/channel-activity",
            button_label="Load Channel Activity",
            chart_title="Channel & Wallet Activity",
            help_text=(
                "Examine transaction types and average wallet activity "
                "across customer segments."
            ),
        )

    with currency_tab:
        st.subheader("Currency & Transaction Status")

        display_visualization(
            key="currency_status",
            endpoint="/visualizations/currency-status",
            button_label="Load Currency & Status",
            chart_title="Currency & Transaction Status",
            help_text=(
                "Compare currency mix and transaction status percentages "
                "across customer segments."
            ),
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Vault Chain Flow Analytics | Powered by Streamlit, FastAPI, "
    "Plotly, and MLflow"
)
