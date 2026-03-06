import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        numeric_df = X.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return self

        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)
            
        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # 1. Missing Values Filter
        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        # 2. High Cardinality Filter (Only for Categorical/Object columns)
        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / len(X_filtered)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)
        
        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        # 3. Correlation Filter (Only for Numeric columns)
        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df['target'] = y
            correlations = temp_df.corr()['target'].abs().drop('target')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes Bitcoin technical indicators from a single 'Close' price column.
    Produces: RSI_14, MACD, MACD_Signal, MACD_Hist, BB_Width, ROC_10, EMA_Ratio
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=["Close"])
        else:
            X_df = X.copy()

        close = X_df["Close"]
        out = pd.DataFrame(index=X_df.index)

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss
        out["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD, Signal Line, Histogram
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        out["MACD"] = macd
        out["MACD_Signal"] = macd_signal
        out["MACD_Hist"] = macd - macd_signal

        # Bollinger Band Width (normalised)
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        out["BB_Width"] = ((ma20 + 2 * std20) - (ma20 - 2 * std20)) / ma20

        # Rate of Change (10)
        out["ROC_10"] = close.pct_change(periods=10) * 100

        # EMA Ratio (fast 20 / slow 50)
        out["EMA_Ratio"] = (
            close.ewm(span=20, adjust=False).mean() /
            close.ewm(span=50, adjust=False).mean()
        )

        return out.dropna()


class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window=60):
        self.window = window
        self.last_beta_ = None
        self.last_alpha_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        if len(X) < self.window:
            raise ValueError(f"Data length {len(X)} is less than window size {self.window}")
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Extractor must be fitted before calling transform.")

        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['price_a', 'price_b'])
        else:
            df = X.copy()
            df.columns = ['price_a', 'price_b']
        
        df[['spread', 'beta']] = self._compute_rolling_regression(df)
        df['z_score'] = self._calculate_z_score(df['spread'])
        df['spread_std'] = df['spread'].rolling(self.window).std()
        df['beta_stability'] = df['beta'].rolling(self.window).std()

        return df

    def _compute_rolling_regression(self, df):
        spreads = np.full(len(df), np.nan)
        betas = np.full(len(df), np.nan)
        
        a_vals = df['price_a'].values
        b_vals = df['price_b'].values

        for i in range(self.window, len(df)):
            y = a_vals[i-self.window:i]
            x = b_vals[i-self.window:i]
            x_with_const = sm.add_constant(x)
            
            model = sm.OLS(y, x_with_const).fit()
            
            alpha, beta = model.params[0], model.params[1]
            betas[i] = beta
            spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)
            
            self.last_alpha_, self.last_beta_ = alpha, beta
            
        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

    def _calculate_z_score(self, spread_series):
        rolling_mean = spread_series.rolling(self.window).mean()
        rolling_std = spread_series.rolling(self.window).std()
        return (spread_series - rolling_mean) / rolling_std
