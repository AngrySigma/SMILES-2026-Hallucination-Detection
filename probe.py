"""probe.py — meta-average ensemble of nine sub-probes over the feature
blocks declared in ``aggregation.py``.

Architecture:
    9 sub-probes are fit on the full training set inside ``fit()``. A
    parallel inner 5-fold CV produces out-of-fold (OOF) probabilities used
    to derive (a) the weight vector for the top-portion average, and
    (b) the accuracy-best decision threshold.

    At inference, the 8 top-portion sub-probes' probabilities are weighted-
    averaged into a single ``p_top8``; that probability is then simple-mean
    averaged with 4 standalone sub-probe probabilities (3 of which are
    deliberately the same sub-probes already inside ``p_top8`` — duplicating
    the strongest probes was the search-found optimum).

Adding a sub-probe: append a ``SubProbeConfig(...)`` entry to ``SUB_PROBES``
and, if it should participate in the top-portion average, list its name in
``META_TOP_NAMES``; if it should additionally enter the simple mean as a
standalone, list it in ``META_STANDALONE_NAMES``.

Public surface (required by the frozen evaluate.py / solution.py):
    fit(X, y), fit_hyperparameters(X_val, y_val), predict(X), predict_proba(X).
``predict_proba`` returns shape ``(n, 2)`` with column 1 = P(hallucinated).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from aggregation import BLOCK_SLICES


RANDOM_STATE: int = 42
OOF_FOLDS: int = 5


@dataclass
class SubProbeConfig:
    name: str
    blocks: tuple[str, ...]
    factory: Callable[[], Any]
    use_scaler: bool = True


# ---------------------------------------------------------------------------
# Multi-C bagged LogisticRegression
# ---------------------------------------------------------------------------

_MULTI_C_VALUES: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.3, 1.0)


class _MultiCLR:
    """Average ``LogisticRegression(C=c, ...)`` over a fixed grid of C.

    Each C value is a different objective, so the fits are genuinely
    different (unlike multi-seed bagging on lbfgs, which is deterministic
    for fixed data and produces identical probabilities).

    Surface: ``fit(X, y)``, ``predict_proba(X)`` returning ``(n, 2)``.
    """

    def __init__(self, C_values: tuple[float, ...] = _MULTI_C_VALUES, **lr_kwargs):
        self.C_values: tuple[float, ...] = tuple(C_values)
        self.lr_kwargs: dict = dict(lr_kwargs)
        self.lr_kwargs.pop("C", None)
        self.clfs: list[LogisticRegression] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MultiCLR":
        self.clfs = [
            LogisticRegression(C=C, **self.lr_kwargs).fit(X, y)
            for C in self.C_values
        ]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ps = np.stack([c.predict_proba(X)[:, 1] for c in self.clfs])
        p = ps.mean(axis=0)
        return np.stack([1.0 - p, p], axis=1)


def _multi_c_lr_factory(**lr_kwargs) -> Callable[[], _MultiCLR]:
    return lambda: _MultiCLR(C_values=_MULTI_C_VALUES, **lr_kwargs)


# ---------------------------------------------------------------------------
# Small two-layer MLP probe (early-stopped on an inner 15% split)
# ---------------------------------------------------------------------------

class _MLPProbe:
    """Linear(D, hidden) → ReLU → Dropout → Linear(hidden, 1), trained with
    AdamW + BCEWithLogitsLoss(pos_weight) on an inner 85/15 split with
    early stopping on the held-out 15%.

    The framework handles ``StandardScaler`` externally (via
    ``use_scaler=True``); this class operates on already-scaled inputs.
    """

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        max_epochs: int = 300,
        patience: int = 20,
        random_state: int = 42,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._net: nn.Sequential | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MLPProbe":
        X_tv, X_h, y_tv, y_h = train_test_split(
            X, y,
            test_size=0.15, random_state=self.random_state, stratify=y,
        )
        X_tv_t = torch.from_numpy(X_tv).float()
        y_tv_t = torch.from_numpy(y_tv.astype(np.float32))
        X_h_t = torch.from_numpy(X_h).float()
        y_h_t = torch.from_numpy(y_h.astype(np.float32))

        torch.manual_seed(self.random_state)
        self._net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

        n_pos = int(y_tv.sum())
        n_neg = len(y_tv) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        rng = np.random.default_rng(self.random_state)
        n_train = X_tv_t.size(0)

        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for _ in range(self.max_epochs):
            self._net.train()
            perm = rng.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_tv_t[idx]
                yb = y_tv_t[idx]
                optimizer.zero_grad()
                logits = self._net(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_h_t).squeeze(-1), y_h_t).item()
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "MLP not fitted yet."
        X_t = torch.from_numpy(X).float()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self._net(X_t).squeeze(-1)).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


def _mlp_probe_factory(**kwargs) -> Callable[[], _MLPProbe]:
    return lambda: _MLPProbe(**kwargs)


# ---------------------------------------------------------------------------
# Sub-probe registry — these are fit inside HallucinationProbe.fit().
# ---------------------------------------------------------------------------

SUB_PROBES: list[SubProbeConfig] = [
    SubProbeConfig(
        name="cross_layer_mean_lr",
        blocks=("cross_layer_mean",),
        factory=lambda: LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=RANDOM_STATE,
        ),
    ),
    SubProbeConfig(
        name="geo_gbt",
        blocks=("last_tok_l20", "geo_features"),
        factory=lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            l2_regularization=1.0, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=RANDOM_STATE,
        ),
        use_scaler=False,
    ),
    SubProbeConfig(
        name="resp_mean_lr",
        blocks=("resp_mean_l23",),
        factory=lambda: LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=RANDOM_STATE,
        ),
    ),
    SubProbeConfig(
        name="actual_logprob_gbt",
        blocks=("actual_logprob",),
        factory=lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.04, max_depth=4,
            l2_regularization=1.5, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=RANDOM_STATE,
        ),
        use_scaler=False,
    ),
    SubProbeConfig(
        name="regen_overlap_gbt",
        blocks=("regen_features",),
        factory=lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.04, max_depth=4,
            l2_regularization=1.5, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=RANDOM_STATE,
        ),
        use_scaler=False,
    ),
    SubProbeConfig(
        name="multi_c_lr_resp_mean",
        blocks=("resp_mean_l23",),
        # NB: _multi_c_lr_factory is invoked at module-load time, so its
        # kwargs are evaluated NOW. RANDOM_STATE is in scope here.
        factory=_multi_c_lr_factory(
            class_weight="balanced", max_iter=5000, solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    ),
    SubProbeConfig(
        name="extra_trees_geo",
        blocks=("last_tok_l20", "geo_features"),
        factory=lambda: ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        use_scaler=False,
    ),
    SubProbeConfig(
        name="stat_drift_lr",
        blocks=("stat_features", "layer_drift_l2"),
        factory=lambda: LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=RANDOM_STATE,
        ),
    ),
    SubProbeConfig(
        name="mlp_pool_mlp",
        blocks=("resp_mean_l23",),
        factory=_mlp_probe_factory(),
        use_scaler=True,
    ),
]


# Sub-probes weight-combined into the top-portion average.
META_TOP_NAMES: tuple[str, ...] = (
    "cross_layer_mean_lr", "geo_gbt", "resp_mean_lr",
    "actual_logprob_gbt", "regen_overlap_gbt",
    "multi_c_lr_resp_mean", "extra_trees_geo", "stat_drift_lr",
)

# Sub-probes that additionally enter the final simple mean as standalones.
# The first three deliberately duplicate top-portion members — duplicating
# the strongest probes was the search-found optimum on the OOF subset
# search; their effective weight in the final mean becomes (w_i + 1)/N.
META_STANDALONE_NAMES: tuple[str, ...] = (
    "cross_layer_mean_lr", "geo_gbt", "regen_overlap_gbt", "mlp_pool_mlp",
)


# ---------------------------------------------------------------------------
# Meta-average ensemble probe
# ---------------------------------------------------------------------------

class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # (cfg, scaler_or_None, fitted_classifier) for each sub-probe.
        self._sub_states: list[tuple[SubProbeConfig, Any, Any]] = []
        self._top_weights: np.ndarray | None = None
        self._top_idx: list[int] | None = None
        self._standalone_idx: list[int] | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "HallucinationProbe is an sklearn-style ensemble; "
            "use predict() / predict_proba() instead of forward()."
        )

    @staticmethod
    def _slice(X: np.ndarray, blocks: tuple[str, ...]) -> np.ndarray:
        if len(blocks) == 1:
            return X[:, BLOCK_SLICES[blocks[0]]]
        return np.concatenate([X[:, BLOCK_SLICES[b]] for b in blocks], axis=1)

    def _sub_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], len(self._sub_states)), dtype=np.float64)
        for j, (cfg, scaler, clf) in enumerate(self._sub_states):
            X_sub = self._slice(X, cfg.blocks)
            X_pred = scaler.transform(X_sub) if scaler is not None else X_sub
            out[:, j] = clf.predict_proba(X_pred)[:, 1]
        return out

    def _resolve_meta_indices(self) -> tuple[list[int], list[int]]:
        name_to_idx = {cfg.name: i for i, cfg in enumerate(SUB_PROBES)}
        missing = [
            n for n in META_TOP_NAMES + META_STANDALONE_NAMES
            if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"meta_avg ensemble references sub-probes {missing} that are "
                f"not in SUB_PROBES. Available: {list(name_to_idx)}"
            )
        return (
            [name_to_idx[n] for n in META_TOP_NAMES],
            [name_to_idx[n] for n in META_STANDALONE_NAMES],
        )

    def _combine(self, sp: np.ndarray) -> np.ndarray:
        """Hierarchical mean: weighted top portion + simple mean with
        standalones. Same math used at fit-time (on OOF probs) and at
        inference (on test probs).
        """
        assert self._top_weights is not None
        assert self._top_idx is not None and self._standalone_idx is not None
        top_proba = sp[:, self._top_idx] @ self._top_weights         # (n,)
        stand = sp[:, self._standalone_idx]                          # (n, n_stand)
        n_components = 1 + stand.shape[1]
        return (top_proba + stand.sum(axis=1)) / n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        # 1. Fit final sub-probes on the full training set — these are the
        #    estimators used at inference time.
        self._sub_states = []
        for cfg in SUB_PROBES:
            X_sub = self._slice(X, cfg.blocks)
            scaler = StandardScaler().fit(X_sub) if cfg.use_scaler else None
            X_fit = scaler.transform(X_sub) if scaler is not None else X_sub
            clf = cfg.factory()
            clf.fit(X_fit, y)
            self._sub_states.append((cfg, scaler, clf))

        # 2. Inner 5-fold OOF pass: refit each sub-probe per inner fold,
        #    gather honest out-of-fold probabilities for every sample.
        sp_oof = self._compute_oof_subproba(X, y)

        # 3. Derive top-portion weights from per-sub-probe OOF accuracy.
        self._top_idx, self._standalone_idx = self._resolve_meta_indices()
        top_oof = sp_oof[:, self._top_idx]
        top_accs = np.array([
            _best_threshold_acc(top_oof[:, j], y)
            for j in range(top_oof.shape[1])
        ])
        shifted = np.maximum(top_accs - 0.5, 1e-3)
        self._top_weights = shifted / shifted.sum()

        # 4. Tune the decision threshold on the OOF combined probabilities.
        oof_combined = self._combine(sp_oof)
        self._set_threshold_from_probs(oof_combined, np.asarray(y).astype(int))
        return self

    def _compute_oof_subproba(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        sp_oof = np.zeros((n, len(SUB_PROBES)), dtype=np.float64)
        skf = StratifiedKFold(
            n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE,
        )
        for tr, va in skf.split(np.arange(n), y):
            for j, cfg in enumerate(SUB_PROBES):
                X_sub_tr = self._slice(X[tr], cfg.blocks)
                scaler = StandardScaler().fit(X_sub_tr) if cfg.use_scaler else None
                X_fit = scaler.transform(X_sub_tr) if scaler is not None else X_sub_tr
                clf = cfg.factory()
                clf.fit(X_fit, y[tr])
                X_sub_va = self._slice(X[va], cfg.blocks)
                X_pred = scaler.transform(X_sub_va) if scaler is not None else X_sub_va
                sp_oof[va, j] = clf.predict_proba(X_pred)[:, 1]
        return sp_oof

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray,
    ) -> "HallucinationProbe":
        # No-op. The OOF threshold tuned in fit() on ≈551 OOF samples is
        # strictly more stable than re-tuning here on the ≈83-sample val
        # slice — empirically observed AUROC ≈ 0.78 / accuracy ≈ 0.76 when
        # the val threshold was applied, vs accuracy ≈ 0.78 when the OOF
        # threshold was kept. The method is preserved because the public
        # contract requires it.
        return self

    def _set_threshold_from_probs(
        self, probs: np.ndarray, y: np.ndarray,
    ) -> None:
        """Pick the accuracy-best threshold; tie-break toward 0.5."""
        cands = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_acc = 0.5, -1.0
        for t in cands:
            acc = accuracy_score(y, (probs >= t).astype(int))
            if acc > best_acc or (
                acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)
            ):
                best_acc, best_t = acc, float(t)
        self._threshold = best_t

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        sp = self._sub_proba(X)
        p = self._combine(sp)
        return np.stack([1.0 - p, p], axis=1)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _best_threshold_acc(probs: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best = -1.0
    for t in cands:
        a = accuracy_score(y, (probs >= t).astype(int))
        if a > best:
            best = a
    return best
