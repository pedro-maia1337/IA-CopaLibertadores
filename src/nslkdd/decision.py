"""Ajuste da regra de decisão para F1-macro usando apenas previsões out-of-fold do treino.

O classificador estima P(classe|x); multiplicar cada probabilidade por um peso
w_c e escolher argmax(w * p) desloca a fronteira a favor das classes raras.
Os pesos são buscados em probabilidades out-of-fold (cross_val_predict no
KDDTrain+), nunca no KDDTest+.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict


def fit_class_weights(proba, y, classes, grid=np.geomspace(0.25, 64, 25), n_rounds=3):
    """Busca coordenada a coordenada pelos pesos que maximizam F1-macro."""
    y = np.asarray(y)
    w = np.ones(len(classes))

    def score(weights):
        return f1_score(y, classes[np.argmax(proba * weights, axis=1)], average="macro")

    best = score(w)
    for _ in range(n_rounds):
        for j in range(len(classes)):
            for g in grid:
                trial = w.copy()
                trial[j] = g
                s = score(trial)
                if s > best + 1e-6:
                    best, w = s, trial
    return w, best


class WeightedDecisionClassifier(BaseEstimator, ClassifierMixin):
    """Envolve um pipeline e aplica pesos de decisão aprendidos via OOF no treino."""

    def __init__(self, estimator, cv=5, n_jobs=-1):
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        oof = cross_val_predict(clone(self.estimator), X, y, cv=self.cv,
                                method="predict_proba", n_jobs=self.n_jobs)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.weights_, self.oof_f1_ = fit_class_weights(oof, y, self.classes_)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X) * self.weights_, axis=1)]
