"""Catálogo de algoritmos e adaptadores para uso direto com rótulos textuais."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from . import config


class EncodedLabelClassifier(BaseEstimator, ClassifierMixin):
    """Codifica os rótulos internamente (necessário para XGBClassifier).

    Ex.: build_pipeline(EncodedLabelClassifier(XGBClassifier(tree_method="hist")))
    Em buscas de hiperparâmetros, use o prefixo model__estimator__.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y), **fit_params)
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


class NativeCategoricalBoost(BaseEstimator, ClassifierMixin):
    """LightGBM/CatBoost com protocol_type, service e flag como categorias nativas.

    Recebe a saída do pré-processador (DataFrame com os categóricos já em códigos ordinais),
    converte essas colunas para inteiros e informa ao modelo quais são categóricas. Codifica
    os rótulos internamente, como EncodedLabelClassifier.
    """

    def __init__(self, estimator, kind):
        self.estimator = estimator
        self.kind = kind

    def _cats(self, X):
        X = X.copy()
        cats = [c for c in config.CATEGORICAL if c in X.columns]
        X[cats] = X[cats].astype(int)
        return X, cats

    def fit(self, X, y):
        X, cats = self._cats(X)
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        key = "categorical_feature" if self.kind == "lgbm" else "cat_features"
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y), **{key: cats})
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._cats(X)[0])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


class SoftVote(BaseEstimator, ClassifierMixin):
    """Média das probabilidades de vários pipelines já montados.

    Cada membro mantém seu próprio pré-processamento. O VotingClassifier do scikit-learn
    não aceita o Pipeline do imbalanced-learn como classificador.
    """

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.members_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.classes_ = self.members_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_model(name, threads=1, **params):
    """Modelo do catálogo com a semente do projeto.

    'HGB_native' trata protocol_type, service e flag como categorias nativas e exige
    service_encoding='ordinal' e categorical_encoding='ordinal' em build_pipeline.
    threads: paralelismo interno; some-o ao das dobras da CV sem passar dos núcleos.
    """
    rs = config.RANDOM_STATE
    if name == "DT":
        return DecisionTreeClassifier(**{"min_samples_leaf": 2, "random_state": rs, **params})
    if name == "RF":
        return RandomForestClassifier(**{"n_estimators": 300, "n_jobs": threads,
                                         "random_state": rs, **params})
    if name == "ET":
        return ExtraTreesClassifier(**{"n_estimators": 300, "n_jobs": threads,
                                       "random_state": rs, **params})
    if name in ("HGB", "HGB_native"):
        native = {"categorical_features": config.CATEGORICAL} if name == "HGB_native" else {}
        return HistGradientBoostingClassifier(**{"max_iter": 300, "learning_rate": 0.06,
                                                 "early_stopping": False, "random_state": rs,
                                                 **native, **params})
    if name == "XGB":
        from xgboost import XGBClassifier

        return EncodedLabelClassifier(XGBClassifier(**{
            "n_estimators": 600, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8,
            "colsample_bytree": 0.7, "tree_method": "hist", "n_jobs": threads,
            "random_state": rs, "eval_metric": "logloss", **params}))
    if name == "LGBM":  # fora da disciplina (boosting por histogramas, crescimento por folha)
        from lightgbm import LGBMClassifier

        return NativeCategoricalBoost(LGBMClassifier(**{
            "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
            "subsample_freq": 1, "colsample_bytree": 0.7, "n_jobs": threads, "random_state": rs,
            "verbose": -1, **params}), "lgbm")
    if name == "CatBoost":  # fora da disciplina (boosting com árvores simétricas)
        from catboost import CatBoostClassifier

        return NativeCategoricalBoost(CatBoostClassifier(**{
            "iterations": 1000, "learning_rate": 0.08, "depth": 6, "thread_count": threads,
            "random_seed": rs, "verbose": 0, "allow_writing_files": False, **params}), "catboost")
    if name == "MLP":
        return MLPClassifier(**{"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "max_iter": 60,
                                "early_stopping": True, "random_state": rs, **params})
    if name == "KNN":
        return KNeighborsClassifier(**{"n_neighbors": 7, "weights": "distance", "n_jobs": threads,
                                       **params})
    if name == "NB":
        return GaussianNB(**params)
    raise ValueError(name)
