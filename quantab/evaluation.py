"""Cross-validation, learning curves, and metrics."""

from typing import Callable
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from tqdm.auto import tqdm

from quantab.quantum_kernels import build_kernel_matrix


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred).statistic)


def evaluate_classical(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """K-fold CV Spearman correlation for a sklearn estimator."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in kf.split(X):
        m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model.__class__()
        m.fit(X[train_idx], y[train_idx])
        scores.append(spearman(y[test_idx], m.predict(X[test_idx])))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}


def evaluate_quantum_kernel(
    kernel_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> dict:
    """K-fold CV for a precomputed quantum kernel SVM."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        K_train = build_kernel_matrix(kernel_fn, X_tr, X_tr)
        K_test = build_kernel_matrix(kernel_fn, X_te, X_tr)
        svr = SVR(kernel="precomputed", C=C)
        svr.fit(K_train, y[train_idx])
        scores.append(spearman(y[test_idx], svr.predict(K_test)))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}


def learning_curve(
    model_or_kernel,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: list[int],
    is_quantum: bool = False,
    n_repeats: int = 3,
    test_size: int = 200,
    random_state: int = 42,
) -> dict:
    """Learning curve: Spearman vs training set size.

    Args:
        model_or_kernel: sklearn estimator or quantum kernel function.
        X: Feature matrix.
        y: Target values.
        train_sizes: List of training set sizes to evaluate.
        is_quantum: If True, treat model_or_kernel as a kernel function.
        n_repeats: Number of random train/test splits per size.
        test_size: Fixed test set size.
        random_state: Base random seed.

    Returns:
        Dict with train_sizes, means, stds.
    """
    rng = np.random.RandomState(random_state)
    results = {s: [] for s in train_sizes}

    for size in tqdm(train_sizes, desc="Learning curve"):
        for rep in range(n_repeats):
            idx = rng.permutation(len(X))
            train_idx = idx[:size]
            test_idx = idx[size : size + test_size]
            if len(test_idx) < 10:
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            if is_quantum:
                K_train = build_kernel_matrix(model_or_kernel, X_tr, X_tr)
                K_test = build_kernel_matrix(model_or_kernel, X_te, X_tr)
                svr = SVR(kernel="precomputed", C=1.0)
                svr.fit(K_train, y_tr)
                y_pred = svr.predict(K_test)
            else:
                import copy
                m = copy.deepcopy(model_or_kernel)
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_te)

            results[size].append(spearman(y_te, y_pred))

    means = [float(np.mean(results[s])) for s in train_sizes]
    stds = [float(np.std(results[s])) for s in train_sizes]
    return {"train_sizes": train_sizes, "means": means, "stds": stds}
