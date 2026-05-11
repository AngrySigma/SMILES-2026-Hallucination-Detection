"""splitting.py — 5-fold stratified CV with a 15% inner stratified val slice.

``split_data`` returns a list of ``(idx_train, idx_val, idx_test)`` tuples.
Indices are non-overlapping and together cover every sample. Stratified at
both levels so the class balance is preserved in every split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for idx_tv, idx_test in skf.split(np.zeros(len(y)), y):
        idx_train, idx_val = train_test_split(
            idx_tv,
            test_size=val_size,
            random_state=random_state,
            stratify=y[idx_tv],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
