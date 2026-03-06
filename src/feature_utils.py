import numpy as np
import pandas as pd
import datetime
import requests
import os
import sys

def extract_features(days=60):
    """
    Fetches Bitcoin historical close prices and computes the 7 technical
    indicators the model was trained on:
    RSI_14, MACD, MACD_Signal, MACD_Hist, BB_Width, ROC_10, EMA_Ratio
    """
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']

    df = pd.DataFrame(prices, columns=['Timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close']].set_index('Date')

    close = df['Close']

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD, Signal Line, Histogram
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd - macd_signal

    # Bollinger Band Width (normalised)
    ma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BB_Width'] = ((ma20 + 2 * std20) - (ma20 - 2 * std20)) / ma20

    # Rate of Change (10)
    df['ROC_10'] = close.pct_change(periods=10) * 100

    # EMA Ratio (fast 20 / slow 50)
    df['EMA_Ratio'] = (
        close.ewm(span=20, adjust=False).mean() /
        close.ewm(span=50, adjust=False).mean()
    )

    features = df.drop(columns=['Close']).dropna()
    return features


def get_bitcoin_historical_prices(days=60):

    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']

    df = pd.DataFrame(prices, columns=['Timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close']].set_index('Date')
    return df
