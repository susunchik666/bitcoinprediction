from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(frozen=True, slots=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class PDTClassifier:
    """
    Lightweight PDT-like ensemble:
    - multiple trees
    - each tree trained on a permuted bootstrap sample (stability)
    - proba = average of predict_proba
    """

    def __init__(self, n_estimators: int = 25, max_depth: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: list[DecisionTreeClassifier] = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PDTClassifier":
        self.trees = []
        n = len(X)
        for i in range(self.n_estimators):
            idx = self._rng.integers(0, n, size=n)  # bootstrap
            # permutation "twist": permute rows inside bootstrap (stability / de-correlation)
            perm = self._rng.permutation(len(idx))
            idx = idx[perm]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + i,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.stack([t.predict_proba(X) for t in self.trees], axis=0)
        return probs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class PDTRegressor:
    def __init__(self, n_estimators: int = 25, max_depth: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: list[DecisionTreeRegressor] = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PDTRegressor":
        self.trees = []
        n = len(X)
        for i in range(self.n_estimators):
            idx = self._rng.integers(0, n, size=n)
            perm = self._rng.permutation(len(idx))
            idx = idx[perm]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + i,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([t.predict(X) for t in self.trees], axis=0)
        return preds.mean(axis=0)