"""Ajuste do limiar de decisão para F1-macro, usando só previsões out-of-fold do treino.

Em problema binário desbalanceado, 0,5 raramente é o melhor corte para F1-macro. O limiar
é buscado em probabilidades out-of-fold e nunca no teste.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .data import duplicate_groups

GRID = np.linspace(0.01, 0.99, 99)


def macro_f1_curve(proba, is_positive, thresholds=GRID):
    """F1-macro binário em cada limiar, vetorizado (sem chamar f1_score na grade)."""
    pos = np.sort(proba[is_positive])
    neg = np.sort(proba[~is_positive])
    fn = np.searchsorted(pos, thresholds, side="left")
    tn = np.searchsorted(neg, thresholds, side="left")
    tp, fp = len(pos) - fn, len(neg) - tn
    return (tp / np.maximum(2 * tp + fp + fn, 1) + tn / np.maximum(2 * tn + fp + fn, 1))


def best_threshold(proba, is_positive, thresholds=GRID, window=5):
    """Limiar de maior F1-macro; a média móvel evita escolher um pico isolado da curva."""
    curve = macro_f1_curve(proba, is_positive, thresholds)
    smooth = np.convolve(curve, np.ones(window) / window, mode="same")
    margin = window // 2
    return float(thresholds[margin + int(np.argmax(smooth[margin:-margin]))])


def positive_proba(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(config.POSITIVE)]


def _fit_predict(pipe, X, y, train, val):
    model = clone(pipe).fit(X.iloc[train], y.iloc[train])
    return val, positive_proba(model, X.iloc[val])


def cross_val_proba(pipe, X, y, groups, n_splits=5, seed=config.RANDOM_STATE, n_jobs=-1):
    """P(anomaly) out-of-fold e o fold de cada linha. Cópias nunca são separadas."""
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups))
    done = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(pipe, X, y, tr, va) for tr, va in splits)
    proba, fold = np.zeros(len(X)), np.zeros(len(X), dtype=int)
    for k, (val, p) in enumerate(done):
        proba[val], fold[val] = p, k
    return proba, fold


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Envolve um pipeline e aprende o limiar de decisão via OOF no próprio conjunto de ajuste."""

    def __init__(self, estimator, n_splits=5, n_jobs=-1):
        self.estimator = estimator
        self.n_splits = n_splits
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        groups = duplicate_groups(X)
        oof, _ = cross_val_proba(self.estimator, X, y, groups, self.n_splits, n_jobs=self.n_jobs)
        # Todas as linhas, como a métrica oficial: o limiar maximiza o F1-macro que será medido.
        self.threshold_ = best_threshold(oof, y.to_numpy() == config.POSITIVE)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        negative = next(c for c in self.classes_ if c != config.POSITIVE)
        return np.where(positive_proba(self, X) >= self.threshold_, config.POSITIVE, negative)
