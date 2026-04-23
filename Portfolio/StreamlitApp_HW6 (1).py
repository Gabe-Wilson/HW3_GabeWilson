"""
Streamlit app for the Sentiment → Signal model.

Input features (must match training notebook):
    ['ADBE', 'MSFT', 'JPM', 'sentiment_textblob']

Output: BUY / HOLD / SELL signal
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline

import shap
from joblib import dump, load

# ── Configuration ────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Access AWS secrets from Streamlit's secrets.toml
aws_id      = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret  = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token   = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket  = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── Model configuration ──────────────────────────────────────────────────────
# These feature names must exactly match the columns used during training
FEATURE_KEYS = ['ADBE', 'MSFT', 'JPM', 'sentiment_textblob']

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_sentiment.shap",
    "pipeline":  "finalized_sentiment_model.tar.gz",
    "keys":      FEATURE_KEYS,
    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in FEATURE_KEYS
    ],
}

# ── AWS session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1',
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    """Download and extract the .tar.gz pipeline from S3, then load it."""
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}",
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(f"{joblib_file}")


def load_shap_explainer(_session, bucket, key, local_path):
    """Download the SHAP explainer from S3 (cached locally)."""
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)


# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df: pd.DataFrame):
    """Send data to the SageMaker endpoint and return the signal label."""
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw_pred  = predictor.predict(input_df.values.astype(np.float32))
        pred_val  = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping   = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        label     = mapping.get(pred_val, str(pred_val))
        return label, 200
    except Exception as exc:
        return f"Error: {exc}", 500


# ── SHAP explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df: pd.DataFrame, session, bucket: str):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer      = load_shap_explainer(
        session, bucket,
        posixpath.join("explainer", explainer_name),
        local_path,
    )
    best_pipeline  = load_pipeline(session, bucket, "sklearn-pipeline-deployment")

    # Apply preprocessing steps only (exclude sampler + model = last 2 steps)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_transformed      = preprocessing_pipeline.transform(input_df)

    try:
        feature_names = best_pipeline[:-2].get_feature_names_out()
    except Exception:
        feature_names = FEATURE_KEYS

    input_df_transformed = pd.DataFrame(input_transformed, columns=feature_names)
    shap_values          = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0])
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 0].values, index=shap_values[0, :, 0].feature_names)
        .abs()
        .idxmax()
    )
    st.info(
        f"**Business Insight:** The most influential factor in this decision was "
        f"**{top_feature}**."
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment → Signal Predictor", layout="wide")
st.title("👨‍💻 Sentiment → Signal Predictor (BUY / HOLD / SELL)")

st.markdown(
    """
    This app uses a machine-learning pipeline trained on news-headline sentiment scores
    to predict the next-day trading signal for the selected stock ticker.

    **Input features:**
    - **ADBE / MSFT / JPM** – predicted sentiment probability from other tickers' headlines
    - **sentiment_textblob** – TextBlob sentiment score for the target ticker's headlines
    """
)

with st.form("pred_form"):
    st.subheader("Input Sentiment Scores")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        label = inp["name"].replace("_", " ").upper()
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                label,
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    res, status = call_model_api(input_df)

    if status == 200:
        color_map = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
        color     = color_map.get(res, "black")
        st.markdown(
            f"<h2 style='color:{color};'>Prediction: {res}</h2>",
            unsafe_allow_html=True,
        )
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
