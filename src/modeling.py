from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


class QuantileClippedRegressor(RegressorMixin, BaseEstimator):
    """Clip predictions to target bounds learned from the training data only."""

    def __init__(self, estimator, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.estimator = estimator
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y):
        target = np.asarray(y, dtype=float)
        self.lower_bound_ = float(np.quantile(target, self.lower_quantile))
        self.upper_bound_ = float(np.quantile(target, self.upper_quantile))
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_", "lower_bound_", "upper_bound_"])
        predictions = self.estimator_.predict(X)
        return np.clip(predictions, self.lower_bound_, self.upper_bound_)
