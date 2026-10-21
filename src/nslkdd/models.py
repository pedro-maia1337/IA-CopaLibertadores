"""Adaptadores de modelos para uso direto com rótulos textuais (normal, DoS, ...)."""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


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
