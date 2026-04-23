import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew
from gensim.models import Word2Vec


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

        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / len(X_filtered)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)

        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

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

    def __init__(self, windows=[5, 10, 20]):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        data = X_df.squeeze()
        X_out = pd.DataFrame(index=X_df.index)

        for w in self.windows:
            X_out[f'EMA_{w}'] = data.ewm(span=w, min_periods=w).mean()

            M = data.diff(w - 1)
            N = data.shift(w - 1)
            X_out[f'ROC_{w}'] = (M / N) * 100

            X_out[f'MOM_{w}'] = data.diff(w)

            delta = data.diff()
            u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
            d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
            avg_gain = u.ewm(com=w - 1, adjust=False).mean()
            avg_loss = d.ewm(com=w - 1, adjust=False).mean()
            rs = avg_gain / avg_loss
            X_out[f'RSI_{w}'] = 100 - (100 / (1 + rs))

            X_out[f'MA_{w}'] = data.rolling(w, min_periods=w).mean()

        return X_out


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
        betas   = np.full(len(df), np.nan)
        a_vals  = df['price_a'].values
        b_vals  = df['price_b'].values

        for i in range(self.window, len(df)):
            y = a_vals[i - self.window:i]
            x = b_vals[i - self.window:i]
            x_with_const = sm.add_constant(x)
            model = sm.OLS(y, x_with_const).fit()
            alpha, beta = model.params[0], model.params[1]
            betas[i]   = beta
            spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)
            self.last_alpha_, self.last_beta_ = alpha, beta

        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

    def _calculate_z_score(self, spread_series):
        rolling_mean = spread_series.rolling(self.window).mean()
        rolling_std  = spread_series.rolling(self.window).std()
        return (spread_series - rolling_mean) / rolling_std


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Converts text (single-column 2-D array / DataFrame) into mean Word2Vec embeddings.
    Works inside a scikit-learn / imbalanced-learn Pipeline.
    """

    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, X, y=None):
        sentences = [str(row[0]).split() for row in X]
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count
        )
        return self

    def transform(self, X):
        def get_mean_vector(text):
            words = str(text).split()
            vectors = [self.model.wv[w] for w in words if w in self.model.wv]
            if not vectors:
                return np.zeros(self.vector_size)
            return np.mean(vectors, axis=0)

        return np.array([get_mean_vector(row[0]) for row in X])
