"""CNN and BiLSTM regressors on per-position LM embeddings.

Both models operate on raw per-position token embeddings — the full
(N, max_len, hidden_dim) tensor, bypassing any fixed pooling operation.
The model learns a task-conditioned aggregation, which can in principle
discover CDR-weighted attention rather than treating all positions equally.

Sklearn-compatible API (fit / predict / get_params / set_params) so both
drop straight into the learning_curve harness with dr='none'.

Input format
------------
X : np.ndarray, shape (N, max_len, hidden_dim), float16 or float32
    Padded per-position embeddings. Padding is zeros; lengths are inferred
    from position norms (|x_i| > 1e-4 ⟹ valid position).

Labels are z-scored internally — no pre-normalisation needed.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Length inference from zero-padded arrays
# ---------------------------------------------------------------------------

def _infer_lengths(X: np.ndarray) -> np.ndarray:
    """Return true sequence lengths by detecting zero-padded positions.

    A position is considered valid if its L1 norm across hidden_dim exceeds
    a small threshold (LM embeddings are never exactly all-zero).
    """
    norms = np.abs(X).sum(axis=-1)          # (N, max_len)
    lengths = (norms > 1e-4).sum(axis=1)    # (N,)
    return np.clip(lengths, 1, X.shape[1]).astype(np.int64)


# ---------------------------------------------------------------------------
# Internal PyTorch modules
# ---------------------------------------------------------------------------

class _CNNModule(nn.Module):
    """Multi-scale 1D-CNN with global max pooling.

    Three parallel convolutions with kernel sizes (3, 5, 7) capture local
    motifs of different lengths. Global max pooling picks the strongest
    activation in each filter regardless of position — ideal for CDR motifs
    that appear at variable offsets in the padded sequence.
    """

    def __init__(self, hidden_dim: int, n_filters: int, kernel_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, n_filters, k, padding="same")
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim) → (batch, hidden_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)

        # Mask padding positions so they don't pollute max pooling
        seq_len = x.size(-1)
        mask = (torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
        # Expand mask for broadcasting with (batch, n_filters, seq_len)
        mask = mask.unsqueeze(1)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))                         # (batch, n_filters, seq_len)
            h = h.masked_fill(~mask, float("-inf"))     # blank padding before max
            h = h.max(dim=-1).values                    # global max: (batch, n_filters)
            pooled.append(h)

        out = torch.cat(pooled, dim=-1)                 # (batch, n_filters * n_kernels)
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)                 # (batch,)


class _BiLSTMModule(nn.Module):
    """Bidirectional LSTM with mean pooling over valid positions.

    The BiLSTM reads the full sequence in both directions; mean pooling over
    non-padding positions produces a fixed-size summary that naturally weights
    residues by how many surrounding context positions they participate in.
    """

    def __init__(self, hidden_dim: int, lstm_hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)  # ×2 for bidirectional

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack to skip padding during LSTM computation
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)  # (batch, max_valid, 2*H)

        # Mean pool over valid positions
        seq_len = out.size(1)
        mask = (
            torch.arange(seq_len, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).float().unsqueeze(-1)                                       # (batch, seq_len, 1)
        pooled = (out * mask).sum(1) / lengths.float().unsqueeze(-1) # (batch, 2*H)

        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers
# ---------------------------------------------------------------------------

class _SequenceRegressor:
    """Base class for per-position sequence regressors."""

    def __init__(
        self,
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.3,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Set after fit:
        self.model_: nn.Module | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0

    def _build_module(self, hidden_dim: int) -> nn.Module:
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "dropout": self.dropout,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "_SequenceRegressor":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SequenceRegressor":
        """Train on per-position embeddings.

        Args:
            X: Shape (N, max_len, hidden_dim). Padded with zeros.
            y: Affinity labels (affinity_norm), shape (N,).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) + 1e-8
        y_norm = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        hidden_dim = X.shape[-1]
        self.model_ = self._build_module(hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        lengths = torch.tensor(_infer_lengths(X), dtype=torch.long)
        X_t = torch.tensor(X.astype(np.float32))
        y_t = torch.tensor(y_norm)
        N = len(X)

        self.model_.train()
        for _ in tqdm(range(self.n_epochs), desc=type(self).__name__, leave=False):
            perm = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_t[idx].to(self.device)
                yb = y_t[idx].to(self.device)
                lb = lengths[idx].to(self.device)

                optimizer.zero_grad()
                pred = self.model_(xb, lb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        lengths = torch.tensor(_infer_lengths(X), dtype=torch.long)
        X_t = torch.tensor(X.astype(np.float32))
        preds: list[np.ndarray] = []

        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = X_t[start : start + self.batch_size].to(self.device)
                lb = lengths[start : start + self.batch_size].to(self.device)
                preds.append(self.model_(xb, lb).cpu().numpy())

        return np.concatenate(preds) * self.y_std_ + self.y_mean_


class CNNRegressor(_SequenceRegressor):
    """1D-CNN regressor on per-position LM embeddings.

    Multi-scale convolutions (kernel sizes 3, 5, 7) capture CDR motifs of
    varying lengths; global max pooling selects the strongest activation per
    filter. Should outperform fixed mean/CLS pooling when positional structure
    is informative for binding — which CDR loops almost certainly are.

    Args:
        n_filters: Convolutional filters per kernel size.
        kernel_sizes: Tuple of kernel widths (motif lengths to probe).
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        dropout: Dropout rate before the output linear layer.
        random_state: RNG seed.
    """

    def __init__(
        self,
        n_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.3,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(n_epochs, batch_size, lr, dropout, random_state, device)
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes

    def get_params(self, deep: bool = True) -> dict:
        return {**super().get_params(), "n_filters": self.n_filters, "kernel_sizes": self.kernel_sizes}

    def _build_module(self, hidden_dim: int) -> nn.Module:
        return _CNNModule(hidden_dim, self.n_filters, self.kernel_sizes, self.dropout)


class BiLSTMRegressor(_SequenceRegressor):
    """Bidirectional LSTM regressor on per-position LM embeddings.

    Reads the full token sequence in both directions; mean pools over valid
    positions to produce a fixed-size representation. Captures long-range
    dependencies (e.g. VH–VL interface) that local CNN kernels might miss.

    Args:
        lstm_hidden: Hidden units per direction (output is 2× this).
        n_layers: Stacked LSTM layers.
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        dropout: Dropout between LSTM layers and before the output linear.
        random_state: RNG seed.
    """

    def __init__(
        self,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.3,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(n_epochs, batch_size, lr, dropout, random_state, device)
        self.lstm_hidden = lstm_hidden
        self.n_layers = n_layers

    def get_params(self, deep: bool = True) -> dict:
        return {**super().get_params(), "lstm_hidden": self.lstm_hidden, "n_layers": self.n_layers}

    def _build_module(self, hidden_dim: int) -> nn.Module:
        return _BiLSTMModule(hidden_dim, self.lstm_hidden, self.n_layers, self.dropout)
