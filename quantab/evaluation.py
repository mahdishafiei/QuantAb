"""Cross-validation, learning curves, and metrics."""

from typing import Callable
import copy
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from tqdm.auto import tqdm

from quantab.quantum_kernels import build_kernel_matrix


def _reduce_fold(
    X_raw_tr: np.ndarray,
    X_raw_te: np.ndarray,
    y_tr: np.ndarray,
    dr: str,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit DR on training data, transform train + test. Never sees test labels."""
    if dr == "pca":
        from sklearn.decomposition import PCA
        m = PCA(n_components=n_components, random_state=42)
        return m.fit_transform(X_raw_tr), m.transform(X_raw_te)
    if dr == "plsr":
        from quantab.embeddings import fit_plsr
        m = fit_plsr(X_raw_tr, y_tr, n_components)
        return m.transform(X_raw_tr), m.transform(X_raw_te)
    raise ValueError(f"Unknown dr: {dr!r}. Choose 'pca', 'plsr', or 'none'.")


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
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray,
    train_sizes: list[int],
    is_quantum: bool = False,
    n_repeats: int = 3,
    test_size: int = 200,
    random_state: int = 42,
    dr: str = "none",
    n_components: int = 10,
) -> dict:
    """Learning curve: Spearman ρ vs training set size.

    Args:
        model_or_kernel: sklearn estimator or quantum kernel function.
        X: Feature matrix (pre-reduced ndarray when dr='none') OR raw embeddings.
           For two-register experiments pass a list of two raw arrays
           [X_ab_raw, X_tgt_raw]; DR is applied per modality then concatenated.
        y: Target values (affinity_norm).
        train_sizes: Training set sizes to evaluate.
        is_quantum: If True, treat model_or_kernel as a kernel function → SVR.
        n_repeats: Random splits per training size.
        test_size: Fixed held-out test size.
        random_state: Base RNG seed.
        dr: Dimensionality reduction applied per fold on training data.
            'none'  — X is already reduced (backward-compatible default).
            'pca'   — unsupervised; negative control.
            'plsr'  — supervised; primary method (fits on y_train only).
        n_components: Components per modality when dr != 'none'.

    Returns:
        Dict with keys train_sizes, means, stds.
    """
    two_modality = isinstance(X, list)
    rng = np.random.RandomState(random_state)
    results = {s: [] for s in train_sizes}
    N = len(X[0]) if two_modality else len(X)

    for size in tqdm(train_sizes, desc="Learning curve"):
        for _ in range(n_repeats):
            idx = rng.permutation(N)
            train_idx = idx[:size]
            test_idx = idx[size : size + test_size]
            if len(test_idx) < 10:
                continue

            y_tr, y_te = y[train_idx], y[test_idx]

            if dr == "none":
                # X is already reduced — use directly
                X_tr = X[train_idx]
                X_te = X[test_idx]
            else:
                if two_modality:
                    # Fit DR on each modality separately, then concatenate
                    tr_parts, te_parts = [], []
                    for X_mod in X:
                        tr, te = _reduce_fold(
                            X_mod[train_idx], X_mod[test_idx], y_tr, dr, n_components
                        )
                        tr_parts.append(tr)
                        te_parts.append(te)
                    X_tr = np.hstack(tr_parts)
                    X_te = np.hstack(te_parts)
                else:
                    X_tr, X_te = _reduce_fold(X[train_idx], X[test_idx], y_tr, dr, n_components)

            if is_quantum:
                K_train = build_kernel_matrix(model_or_kernel, X_tr, X_tr)
                K_test = build_kernel_matrix(model_or_kernel, X_te, X_tr)
                svr = SVR(kernel="precomputed", C=1.0)
                svr.fit(K_train, y_tr)
                y_pred = svr.predict(K_test)
            else:
                m = copy.deepcopy(model_or_kernel)
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_te)

            results[size].append(spearman(y_te, y_pred))

    means = [float(np.mean(results[s])) for s in train_sizes]
    stds = [float(np.std(results[s])) for s in train_sizes]
    return {"train_sizes": train_sizes, "means": means, "stds": stds}
