"""
Reusable feature engineering transformers and custom balanced/ordinal estimators
inspired by model-selection recommendations for Spotify Over Time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

ERA_ORDER = ["<2000", "2000-2010", "2011-2020", "2021+"]

RAW_NUMERIC_FEATURES = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
    "tempo",
]

ENGINEERED_NUMERIC_FEATURES = [
    "energy_x_loudness",
    "loudness_per_acousticness",
    "energy_per_acousticness",
    "valence_x_danceability",
    "acoustic_x_instrumental",
    "loudness_per_energy",
]

ALL_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["key"]
BINARY_FEATURES = ["mode"]
RAW_INPUT_FEATURES = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
ALL_MODEL_FEATURES = ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


class AudioFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Appends domain-specific audio interaction features while preserving the 11
    original input columns so pipelines still accept raw song feature rows.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "AudioFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=RAW_INPUT_FEATURES)
        eps = 0.05
        df["energy_x_loudness"] = df["energy"] * df["loudness"]
        df["loudness_per_acousticness"] = df["loudness"] / (df["acousticness"] + eps)
        df["energy_per_acousticness"] = df["energy"] / (df["acousticness"] + eps)
        df["valence_x_danceability"] = df["valence"] * df["danceability"]
        df["acoustic_x_instrumental"] = df["acousticness"] * df["instrumentalness"]
        df["loudness_per_energy"] = df["loudness"] / (df["energy"] + eps)
        return df[ALL_MODEL_FEATURES]


class OrdinalGBDTClassifier(BaseEstimator, ClassifierMixin):
    """
    Frank & Hall (2001) cumulative ordinal classifier built on sample-weighted
    Histogram-based Gradient Boosted Decision Trees.

    Exploits the natural chronological ordering `<2000` < `2000-2010` < `2011-2020` < `2021+`
    by fitting K-1 binary models for P(era > k) using balanced sample weights.
    """

    def __init__(
        self,
        era_order: tuple[str, ...] = tuple(ERA_ORDER),
        categorical_features: tuple[str, ...] = ("key",),
        learning_rate: float = 0.05,
        max_iter: int = 400,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.era_order = era_order
        self.categorical_features = categorical_features
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OrdinalGBDTClassifier":
        from scipy.optimize import minimize
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.model_selection import StratifiedShuffleSplit

        self.classes_ = np.unique(y)
        era_to_rank = {era: idx for idx, era in enumerate(self.era_order)}
        y_rank = pd.Series(y).map(era_to_rank).to_numpy()

        # Carve out a 15% internal validation split from X_train to tune monotonic era thresholds
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
        tr_idx, val_idx = next(sss.split(X, y_rank))
        X_tr, y_tr = X.iloc[tr_idx], y_rank[tr_idx]
        X_val, y_val = X.iloc[val_idx], y_rank[val_idx]

        sw_tr = compute_sample_weight(class_weight="balanced", y=y_tr)
        val_estimators: list[HistGradientBoostingClassifier] = []
        for k in range(len(self.era_order) - 1):
            clf_val = HistGradientBoostingClassifier(
                categorical_features=list(self.categorical_features),
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state + k,
            )
            clf_val.fit(X_tr, (y_tr > k).astype(int), sample_weight=sw_tr)
            val_estimators.append(clf_val)

        # Expected era score E[Y|X] = sum_{k=0}^{K-2} P(Y > k | X) in [0, 3]
        val_score = sum(est.predict_proba(X_val)[:, 1] for est in val_estimators)

        def _obj(th: np.ndarray) -> float:
            t0, t1, t2 = np.sort(th)
            if t1 - t0 < 0.15 or t2 - t1 < 0.15:
                return 10.0
            preds = np.digitize(val_score, [t0, t1, t2])
            return -(
                0.55 * balanced_accuracy_score(y_val, preds)
                + 0.45 * f1_score(y_val, preds, average="macro")
            )

        opt = minimize(_obj, x0=np.array([0.85, 1.50, 2.15]), method="Nelder-Mead")
        self.thresholds_ = np.sort(opt.x)

        # Refit cumulative estimators on 100% of training data
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_rank)
        self.estimators_: list[HistGradientBoostingClassifier] = []
        for k in range(len(self.era_order) - 1):
            y_bin = (y_rank > k).astype(int)
            clf = HistGradientBoostingClassifier(
                categorical_features=list(self.categorical_features),
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state + k,
            )
            clf.fit(X, y_bin, sample_weight=sample_weights)
            self.estimators_.append(clf)

        return self

    def expected_era_score(self, X: pd.DataFrame) -> np.ndarray:
        """Continuous ordinal era score E[era_rank | X] in [0, K-1]."""
        return sum(est.predict_proba(X)[:, 1] for est in self.estimators_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Cumulative probabilities P(era > k) for k = 0, 1, ..., K-2
        cum_gt = [est.predict_proba(X)[:, 1] for est in self.estimators_]

        K = len(self.era_order)
        ord_probs = np.zeros((len(X), K), dtype=float)
        ord_probs[:, 0] = 1.0 - cum_gt[0]
        for k in range(1, K - 1):
            ord_probs[:, k] = np.clip(cum_gt[k - 1] - cum_gt[k], 1e-6, 1.0)
        ord_probs[:, K - 1] = cum_gt[K - 2]

        ord_probs = np.clip(ord_probs, 1e-6, 1.0)
        ord_probs /= ord_probs.sum(axis=1, keepdims=True)

        # Align column order with self.classes_ (lexicographical, matching sklearn convention)
        era_to_ord_col = {era: idx for idx, era in enumerate(self.era_order)}
        aligned = np.column_stack([ord_probs[:, era_to_ord_col[c]] for c in self.classes_])
        return aligned

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.expected_era_score(X)
        pred_ranks = np.digitize(scores, self.thresholds_)
        return np.asarray(self.era_order)[pred_ranks]


class BalancedMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-Layer Perceptron benchmark with class-balanced stratified resampling
    (since sklearn's MLPClassifier does not natively accept `class_weight`).
    Encodes string labels to integer indices to avoid NumPy 2.0 object-dtype
    `isnan` issues during sklearn early stopping.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        alpha: float = 1e-2,
        batch_size: int = 512,
        learning_rate_init: float = 1e-3,
        max_iter: int = 50,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: pd.Series) -> "BalancedMLPClassifier":
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        y_int = np.searchsorted(self.classes_, y_arr)
        rng = np.random.default_rng(self.random_state)

        per_class_n = len(y_int) // len(self.classes_)
        sampled_indices = []
        for cls_id in range(len(self.classes_)):
            cls_idx = np.flatnonzero(y_int == cls_id)
            replace = len(cls_idx) < per_class_n
            chosen = rng.choice(cls_idx, size=per_class_n, replace=replace)
            sampled_indices.append(chosen)

        bal_idx = np.concatenate(sampled_indices)
        rng.shuffle(bal_idx)

        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            n_iter_no_change=6,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.mlp_.fit(X[bal_idx], y_int[bal_idx])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.mlp_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_int = self.mlp_.predict(X)
        return self.classes_[pred_int]

