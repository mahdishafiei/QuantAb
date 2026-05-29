"""Classical baselines using scikit-learn.

Headline predictor is a small MLP head on the frozen, PCA-reduced features —
matching the downstream head the lab uses on top of LM embeddings. The kernel
SVMs are retained as the apples-to-apples comparator for the quantum kernel
(which feeds a precomputed-kernel SVR).
"""

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def mlp() -> Pipeline:
    """Small MLP head on standardized features (lab-style downstream predictor)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=500,
            random_state=42,
        )),
    ])


def linear_svm() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=1.0))])


def rbf_svm() -> Pipeline:
    """Classical kernel comparator for the quantum kernel."""
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, gamma="scale"))])


def random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


MODELS = {
    "mlp": mlp,
    "rbf_svm": rbf_svm,
    "linear_svm": linear_svm,
    "random_forest": random_forest,
}
