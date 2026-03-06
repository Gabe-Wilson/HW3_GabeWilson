import os, sys, warnings
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
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline

import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )


session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_prices = get_bitcoin_historical_prices()

MIN_VAL = 0.5 * df_prices["Close"].min()
MAX_VAL = 2.0 * df_prices["Close"].max()
DEFAULT_VAL = df_prices["Close"].mean()

FEATURE_COLS = ["RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Width", "ROC_10", "EMA_Ratio"]

MODEL_INFO = {
    "endpoint": aws_endpoint_bitcoin,
    "explainer": 'explainer_bitcoin.shap',
    "pipeline": 'finalized_bitcoin_model.tar.gz',
    "inputs": [{"name": "Close Price", "type": "number", "min": MIN_VAL, "max": MAX_VAL, "default": DEFAULT_VAL, "step": 100.0}]
}


# Feature Engineering
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger_band_width(series, window=20, num_std=2):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / ma

def compute_roc(series, period=10):
    return series.pct_change(periods=period) * 100

def compute_ema_ratio(series, fast=20, slow=50):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast / ema_slow

def build_feature_row(df_prices, new_close_price):
    extended = pd.concat([
        df_prices[["Close"]],
        pd.DataFrame([{"Close": new_close_price}])
    ], ignore_index=True)

    extended["RSI_14"] = compute_rsi(extended["Close"], period=14)
    macd, sig, hist = compute_macd(extended["Close"])
    extended["MACD"] = macd
    extended["MACD_Signal"] = sig
    extended["MACD_Hist"] = hist
    extended["BB_Width"] = compute_bollinger_band_width(extended["Close"])
    extended["ROC_10"] = compute_roc(extended["Close"], period=10)
    extended["EMA_Ratio"] = compute_ema_ratio(extended["Close"])

    return extended[FEATURE_COLS].dropna()


# Model Loading
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df.values)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(pred_val, pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    full_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Rebuild preprocessing pipeline (no SMOTE, no model)
    preproc_steps = [
        (name, step) for name, step in full_pipeline.steps
        if name not in ('sampler', 'model')
    ]
    preprocessing_pipeline = Pipeline(steps=preproc_steps)
    input_transformed = preprocessing_pipeline.transform(input_df)

    # Get selected feature names from SelectKBest mask
    selector = full_pipeline.named_steps.get("feature_selection")
    if selector is not None:
        feature_names = np.array(FEATURE_COLS)[selector.get_support()]
    else:
        feature_names = np.array(FEATURE_COLS)

    shap_values = explainer(input_transformed)

    exp = shap.Explanation(
        values=shap_values[-1, :, 0],
        base_values=explainer.expected_value[0],
        data=input_transformed[-1],
        feature_names=list(feature_names)
    )

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(exp)
    st.pyplot(fig)

    top_feature = pd.Series(exp.values, index=exp.feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment Compiler", layout="wide")
st.title("👨‍💻 ML Deployment Compiler")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'],
                value=inp['default'], step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_df = build_feature_row(df_prices, user_inputs["Close Price"])

    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)



