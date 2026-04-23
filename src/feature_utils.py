"""
feature_utils.py
----------------
Feature utilities for the HW6 Option 3 Streamlit app.

The deployed XGBoost signal model was trained on daily average
predicted-sentiment scores, one column per peer ticker of the target (MSFT).
The feature columns used at training time are:
    ['ADBE', 'MSFT', 'JPM', 'sentiment_textblob']

This file provides:
  - FEATURE_KEYS  : the ordered list of feature column names
  - extract_features() : returns a one-row DataFrame of neutral defaults
                         so the Streamlit app always has a valid template
                         to build input rows from.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Must exactly match the 'top_features' list printed by the training notebook
# and the FEATURE_KEYS list in StreamlitApp_HW6.py
# ---------------------------------------------------------------------------
FEATURE_KEYS = ['ADBE', 'MSFT', 'JPM', 'sentiment_textblob']


def extract_features() -> pd.DataFrame:
    """
    Return a single-row DataFrame of neutral (zero) sentiment scores.

    The Streamlit app uses this as a template to know which columns exist
    and what sensible default values look like.  All values are 0.0 because
    sentiment scores are centred around zero (range roughly -1 to +1).

    Returns
    -------
    pd.DataFrame
        Shape (1, len(FEATURE_KEYS)), columns = FEATURE_KEYS, all values 0.0
    """
    return pd.DataFrame(
        [[0.0] * len(FEATURE_KEYS)],
        columns=FEATURE_KEYS
    )


def build_input_row(user_inputs: dict) -> pd.DataFrame:
    """
    Convert the Streamlit form's user_inputs dict into a model-ready DataFrame.

    Parameters
    ----------
    user_inputs : dict
        Keys are feature names (must be a subset of FEATURE_KEYS),
        values are floats entered by the user.

    Returns
    -------
    pd.DataFrame
        Shape (1, len(FEATURE_KEYS)) with columns in the correct order.
    """
    row = {k: float(user_inputs.get(k, 0.0)) for k in FEATURE_KEYS}
    return pd.DataFrame([row], columns=FEATURE_KEYS)


def get_feature_metadata() -> list:
    """
    Return UI metadata for each feature so the Streamlit form can be built
    dynamically without hardcoding anything in the app file.

    Returns
    -------
    list of dict, one entry per feature:
        {
            'name'    : str,   # column name / form label
            'min'     : float,
            'max'     : float,
            'default' : float,
            'step'    : float,
            'help'    : str    # tooltip shown in the Streamlit form
        }
    """
    descriptions = {
        'ADBE':               'Predicted sentiment score from ADBE headlines',
        'MSFT':               'Predicted sentiment score from MSFT headlines',
        'JPM':                'Predicted sentiment score from JPM headlines',
        'sentiment_textblob': 'TextBlob polarity score for the target ticker',
    }

    return [
        {
            'name':    key,
            'min':     -1.0,
            'max':      1.0,
            'default':  0.0,
            'step':     0.01,
            'help':    descriptions.get(key, key),
        }
        for key in FEATURE_KEYS
    ]
