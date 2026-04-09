import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json
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

warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features, convert_input_pca_regression

# Access secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session — wrapped in try/except
try:
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
except Exception as e:
    st.error(f"AWS session failed: {e}")
    st.stop()

# FIX 1: Wrap extract_features in try/except so rate limit doesn't crash the app
@st.cache_data(ttl=3600)
def load_features():
    try:
        return extract_features()
    except Exception as e:
        st.warning(f"Could not load live market data: {e}")
        return None

df_features = load_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_pca.shap',
    "pipeline": 'finalized_pca_model.tar.gz',
    "keys": ["HON", "AVY"],
    "inputs": [
        {"name": "HON", "type": "number", "min": 0.0, "default": 100.0, "step": 10.0},
        {"name": "AVY", "type": "number", "min": 0.0, "default": 100.0, "step": 10.0},
    ]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(Filename=filename, Bucket=bucket,
                            Key=f"{key}/{os.path.basename(filename)}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

def call_model_api(input_array):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_array)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_explanation(input_array, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
    shap_values = explainer(input_array)
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)
    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor was **{top_feature}**.")

# UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], value=inp['default'], step=inp['step']
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    try:
        # FIX 2: Convert user inputs to the full feature row the model expects
        input_df = convert_input_pca_regression(
            json.dumps(user_inputs), 'application/json'
        )
        input_array = input_df.values

        res, status = call_model_api(input_array)
        if status == 200:
            st.metric("Prediction Result", res)
            display_explanation(input_array, session, aws_bucket)
        else:
            st.error(res)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
