"""Classical kernel baselines using scikit-learn."""

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def linear_svm() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=1.0))])


def rbf_svm() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, gamma="scale"))])


def poly_svm() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="poly", degree=3, C=1.0))])


def random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


MODELS = {
    "linear_svm": linear_svm,
    "rbf_svm": rbf_svm,
    "poly_svm": poly_svm,
    "random_forest": random_forest,
}
