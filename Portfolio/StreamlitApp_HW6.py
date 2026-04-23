"""
Streamlit app – HW6 Option 3
Predicts the BUY / HOLD / SELL signal for MSFT using peer-ticker
sentiment scores fed into an XGBoost pipeline on AWS SageMaker.
"""

import os
import sys
import warnings
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline

import shap
from joblib import load

warnings.simplefilter("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Import feature utilities ──────────────────────────────────────────────────
from src.feature_utils import FEATURE_KEYS, get_feature_metadata, build_input_row

# ── AWS credentials ───────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_sentiment.shap",
    "pipeline":  "finalized_sentiment_model.tar.gz",
    "keys":      FEATURE_KEYS,
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
    return joblib.load(joblib_file)


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)


# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw_pred = predictor.predict(input_df.values.astype(np.float32))
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping  = {0: "SELL", 1: "HOLD", 2: "BUY"}
        return mapping.get(pred_val, str(pred_val)), 200
    except Exception as exc:
        return f"Error: {exc}", 500


# ── SHAP explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df: pd.DataFrame, session, bucket: str):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer     = load_shap_explainer(
        session, bucket,
        posixpath.join("explainer", explainer_name),
        local_path,
    )
    best_pipeline = load_pipeline(session, bucket, "sklearn-pipeline-deployment")

    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_transformed      = preprocessing_pipeline.transform(input_df)

    try:
        feature_names = best_pipeline[:-2].get_feature_names_out()
    except Exception:
        feature_names = FEATURE_KEYS

    input_df_transformed = pd.DataFrame(input_transformed, columns=feature_names)
    shap_values          = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, _ = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0])
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 0].values,
                  index=shap_values[0, :, 0].feature_names)
        .abs()
        .idxmax()
    )
    st.info(
        f"**Business Insight:** The most influential factor in this decision "
        f"was **{top_feature}**."
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment → Signal Predictor", layout="wide")
st.title("📈 Sentiment → Signal Predictor (BUY / HOLD / SELL)")
st.markdown(
    """
    Enter the predicted sentiment scores for each ticker.
    The model will predict the next-day trading signal for **MSFT**.

    | Feature | Description |
    |---|---|
    | **ADBE / JPM** | Peer-ticker headline sentiment (Word2Vec classifier output) |
    | **MSFT** | MSFT headline sentiment (Word2Vec classifier output) |
    | **sentiment_textblob** | TextBlob polarity score for the target ticker |
    """
)

feature_meta = get_feature_metadata()

with st.form("pred_form"):
    st.subheader("Input Sentiment Scores")
    cols        = st.columns(2)
    user_inputs = {}

    for i, meta in enumerate(feature_meta):
        with cols[i % 2]:
            user_inputs[meta["name"]] = st.number_input(
                meta["name"].replace("_", " ").upper(),
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
                help=meta["help"],
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_df = build_input_row(user_inputs)

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
