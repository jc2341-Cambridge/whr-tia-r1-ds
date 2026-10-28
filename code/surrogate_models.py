from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from physics_constraints import PhysicsConstraintAgent


def _poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [np.ones((x.shape[0], 1))]
    n_features = x.shape[1]
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            cols.append(np.prod(x[:, comb], axis=1, keepdims=True))
    return np.hstack(cols)


@dataclass
class PolynomialRidgeModel:
    degree: int = 3
    alpha: float = 1e-3

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRidgeModel":
        self.x_mean_ = x.mean(axis=0)
        self.x_std_ = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = np.where(y.std(axis=0) == 0, 1.0, y.std(axis=0))

        xs = (x - self.x_mean_) / self.x_std_
        ys = (y - self.y_mean_) / self.y_std_
        phi = _poly_features(xs, self.degree)
        reg = self.alpha * np.eye(phi.shape[1])
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(phi.T @ phi + reg, phi.T @ ys)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.x_mean_) / self.x_std_
        phi = _poly_features(xs, self.degree)
        return (phi @ self.coef_) * self.y_std_ + self.y_mean_


class BootstrapSurrogate:
    """Small multi-output ensemble for prediction and epistemic uncertainty."""

    def __init__(
        self,
        n_estimators: int = 80,
        degree: int = 3,
        alpha: float = 1e-2,
        random_state: int = 42,
        physics_agent: PhysicsConstraintAgent | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.degree = degree
        self.alpha = alpha
        self.random_state = random_state
        self.physics_agent = physics_agent or PhysicsConstraintAgent()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BootstrapSurrogate":
        rng = np.random.default_rng(self.random_state)
        self.models_: list[PolynomialRidgeModel] = []
        n = x.shape[0]
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            model = PolynomialRidgeModel(degree=self.degree, alpha=self.alpha).fit(x[idx], y[idx])
            self.models_.append(model)
        return self

    def predict_members(self, x: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(x) for m in self.models_], axis=0)
        for i in range(preds.shape[0]):
            preds[i] = self.physics_agent.clip_predictions(preds[i])
        return preds

    def predict(self, x: np.ndarray, return_std: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        members = self.predict_members(x)
        mean = members.mean(axis=0)
        if not return_std:
            return mean
        std = members.std(axis=0)
        return mean, std

