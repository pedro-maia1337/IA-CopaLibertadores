"""Blocos de pré-processamento e montagem de pipelines imblearn.

Tudo que aprende parâmetros (imputação, codificação, escala, seleção, balanceamento,
remoção de redundâncias) fica DENTRO do Pipeline, então em validação cruzada é ajustado
só nos folds de treino e nunca enxerga o fold de validação ou o teste.
Ordem: redundância -> domínio -> imputação/codificação/transformação -> redução ->
balanceamento -> modelo.
"""
import numpy as np
import pandas as pd
from imblearn import FunctionSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, QuantileTransformer, RobustScaler,
                                   StandardScaler, TargetEncoder)

from . import config
from .data import conflict_mask, duplicate_groups

# ---------------------------------------------------------------- redundância

def _ordinal_hgb():
    """Modelo auxiliar rápido e tolerante a NaN para decidir rótulos conflitantes."""
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), config.CATEGORICAL)],
        remainder="passthrough")
    return make_pipeline(encode, HistGradientBoostingClassifier(
        max_iter=200, categorical_features=[0, 1, 2], random_state=config.RANDOM_STATE))


def _relabel_conflicts(X, y, keep_clean, conflicts, margin):
    """Rotula conflitos pelo modelo treinado nas linhas limpas; incertos saem do treino."""
    model = _ordinal_hgb().fit(X.iloc[keep_clean], y[keep_clean])
    labels = model.classes_
    proba = model.predict_proba(X.iloc[conflicts])
    certain = proba.max(axis=1) >= 1 - margin
    return labels[proba.argmax(axis=1)], certain


def _majority_labels(groups, y):
    """Rótulo com maioria estrita em cada grupo; empates ficam sem rótulo (NaN)."""
    counts = pd.DataFrame({"g": groups, "y": y}).groupby(["g", "y"]).size().reset_index(name="n")
    top = counts[counts["n"] == counts.groupby("g")["n"].transform("max")]
    winners = top[~top["g"].duplicated(keep=False)].set_index("g")["y"]
    return pd.Series(groups).map(winners)


def _resolve_redundancy(X, y, strategy, margin=0.2):
    y = np.asarray(y)
    groups = duplicate_groups(X)
    first = ~pd.Series(groups).duplicated().to_numpy()
    conflict = conflict_mask(groups, y)
    if strategy == "drop_duplicates":  # remove só cópias com o mesmo rótulo
        keep = ~pd.DataFrame({"g": groups, "y": y}).duplicated().to_numpy()
        return X.iloc[keep], y[keep]
    if strategy == "drop_inconsistent":
        keep = first & ~conflict
        return X.iloc[keep], y[keep]
    if strategy == "majority_label":
        labels = _majority_labels(groups, y)
        keep = first & labels.notna().to_numpy()
        return X.iloc[keep], labels.to_numpy()[keep]
    if strategy == "relabel_model":
        clean = first & ~conflict
        doubtful = np.flatnonzero(first & conflict)
        new_y, certain = _relabel_conflicts(X, y, clean, doubtful, margin)
        rows = np.r_[np.flatnonzero(clean), doubtful[certain]]
        return X.iloc[rows], np.r_[y[clean], new_y[certain]]
    raise ValueError(strategy)


def make_redundancy_sampler(strategy: str = "none"):
    """'none' | 'drop_duplicates' | 'drop_inconsistent' | 'majority_label' | 'relabel_model'."""
    if strategy == "none":
        return None
    return FunctionSampler(func=_resolve_redundancy, validate=False,
                           kw_args={"strategy": strategy})

# ---------------------------------------------------------------- domínio

class DomainCleaner(BaseEstimator, TransformerMixin):
    """Trata valores impossíveis pela definição do atributo; não aprende nada dos dados.

    strategy: 'nan' (vira ausente) | 'clip' (satura no limite) | 'none'.
    Cobre taxas > 1, contagens acima da janela e negativos em atributos não negativos.
    """

    def __init__(self, strategy="nan"):
        self.strategy = strategy

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if self.strategy == "none":
            return X
        if self.strategy not in ("nan", "clip"):
            raise ValueError(self.strategy)
        X = X.copy()
        numeric = X[config.NUMERIC]
        limits = pd.Series(config.DOMAIN_MAX).reindex(config.NUMERIC)
        if self.strategy == "clip":
            X[config.NUMERIC] = numeric.clip(lower=0, upper=limits, axis=1)
        else:
            X[config.NUMERIC] = numeric.mask((numeric < 0) | numeric.gt(limits, axis=1))
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# ---------------------------------------------------------------- engenharia de atributos

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Atributos derivados por linha (razões e interações), sem parâmetros aprendidos.

    Cada coluna nova é função só dos 41 atributos de rede da própria linha, então aplicá-la ao
    teste não usa nenhuma informação dele. Árvores precisam de muitos cortes para aproximar
    produtos e razões; entregá-los prontos pode ajudar. Ausentes propagam como NaN.
    """

    NAMES = ["bytes_ratio_log", "src_bytes_per_sec_log", "zero_bytes", "srv_count_ratio",
             "serror_count", "rerror_count", "same_srv_count", "diff_srv_count",
             "dst_host_srv_ratio", "dst_host_serror_count", "dst_host_rerror_count",
             "dst_host_same_srv_count"]

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        pos = lambda c: X[c].clip(lower=0)  # noqa: E731 (negativos já viram NaN no DomainCleaner)
        X["bytes_ratio_log"] = np.log1p(pos("src_bytes")) - np.log1p(pos("dst_bytes"))
        X["src_bytes_per_sec_log"] = np.log1p(pos("src_bytes") / (pos("duration") + 1))
        X["zero_bytes"] = ((X["src_bytes"] == 0) & (X["dst_bytes"] == 0)).astype(float)
        X["srv_count_ratio"] = pos("srv_count") / (pos("count") + 1)
        X["serror_count"] = X["serror_rate"] * X["count"]
        X["rerror_count"] = X["rerror_rate"] * X["count"]
        X["same_srv_count"] = X["same_srv_rate"] * X["count"]
        X["diff_srv_count"] = X["diff_srv_rate"] * X["count"]
        X["dst_host_srv_ratio"] = pos("dst_host_srv_count") / (pos("dst_host_count") + 1)
        X["dst_host_serror_count"] = X["dst_host_serror_rate"] * X["dst_host_count"]
        X["dst_host_rerror_count"] = X["dst_host_rerror_rate"] * X["dst_host_count"]
        X["dst_host_same_srv_count"] = X["dst_host_same_srv_rate"] * X["dst_host_count"]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.r_[self.feature_names_in_, self.NAMES].astype(object)

# ---------------------------------------------------------------- codificação

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Substitui cada categoria pela sua frequência relativa no treino."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = {c: X[c].value_counts(normalize=True) for c in X.columns}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return np.column_stack([X[c].map(m).astype(float).fillna(0.0).to_numpy()
                                for c, m in self.maps_.items()])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


def make_encoder(kind: str):
    """Codificação simbólico-numérica; 'service' tem 67 valores e os demais, poucos."""
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "onehot_rare":  # agrupa categorias com < 0,1% das linhas em 'infrequent'
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                             sparse_output=False)
    if kind == "frequency":
        return FrequencyEncoder()
    if kind == "target":  # cross-fitting interno evita vazamento do rótulo
        return TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE))
    if kind == "ordinal":  # só faz sentido para árvores
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(kind)


def _symbolic(kind):
    """Ausente em categórico vira a categoria 'ausente', que o teste também apresenta."""
    return make_pipeline(SimpleImputer(strategy="constant", fill_value="ausente"), make_encoder(kind))

# ---------------------------------------------------------------- numéricos

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_scaler(kind: str):
    return {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "quantile": lambda: QuantileTransformer(output_distribution="normal",
                                                random_state=config.RANDOM_STATE),
        "none": lambda: None,
    }[kind]()


def make_imputer(kind: str):
    """'none' (NaN nativo: árvores) | 'mean' | 'median' | 'median_indicator' | 'iterative'."""
    if kind == "none":
        return None
    if kind in ("mean", "median"):
        return SimpleImputer(strategy=kind)
    if kind == "median_indicator":  # preserva o sinal do ausente MNAR (dst_host_serror_rate)
        return SimpleImputer(strategy="median", add_indicator=True)
    if kind == "iterative":
        return IterativeImputer(max_iter=5, n_nearest_features=10, sample_posterior=False,
                                random_state=config.RANDOM_STATE)
    raise ValueError(kind)


def _numeric(columns_log, scaling, imputation):
    steps = [make_imputer(imputation)]
    if columns_log:
        steps.append(FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))
    steps.append(make_scaler(scaling))
    steps = [s for s in steps if s is not None]
    return make_pipeline(*steps) if steps else "passthrough"


def make_preprocessor(service_encoding="onehot", categorical_encoding="onehot", scaling="standard",
                      log_transform=True, imputation="median",
                      drop=tuple(config.CONSTANT_IN_TRAIN), extra=()):
    """extra: colunas numéricas criadas antes (FeatureBuilder), tratadas como as demais."""
    numeric = [c for c in config.NUMERIC if c not in drop] + list(extra)
    heavy = [c for c in config.HEAVY_TAILED if c in numeric] if log_transform else []
    light = [c for c in numeric if c not in heavy]
    transformers = [
        ("cat", _symbolic(categorical_encoding), config.LOW_CARDINALITY),
        ("service", _symbolic(service_encoding), ["service"]),
        ("num", _numeric(False, scaling, imputation), light),
    ]
    if heavy:
        transformers.append(("log", _numeric(True, scaling, imputation), heavy))
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return prep.set_output(transform="pandas")

# ---------------------------------------------------------------- redução

def _mutual_info(X, y):
    return mutual_info_classif(X, y, random_state=config.RANDOM_STATE)


def make_reducer(kind="none", k=20, variance=0.95):
    """'none' | 'pca' | 'kbest_f' | 'kbest_mi' | 'tree_select'."""
    if kind == "none":
        return None
    if kind == "pca":
        return PCA(n_components=variance, random_state=config.RANDOM_STATE)
    if kind == "kbest_f":
        return SelectKBest(f_classif, k=k)
    if kind == "kbest_mi":
        return SelectKBest(_mutual_info, k=k)
    if kind == "tree_select":
        return SelectFromModel(ExtraTreesClassifier(n_estimators=100, n_jobs=1,
                                                    random_state=config.RANDOM_STATE),
                               threshold="median")
    raise ValueError(kind)

# ---------------------------------------------------------------- balanceamento

class OversampleTo:
    """sampling_strategy: eleva cada classe a pelo menos min_count instâncias."""

    def __init__(self, min_count):
        self.min_count = min_count

    def __call__(self, y):
        counts = pd.Series(y).value_counts()
        return {c: int(max(n, self.min_count)) for c, n in counts.items()}


class UndersampleTo:
    """sampling_strategy: reduz cada classe a no máximo max_count instâncias."""

    def __init__(self, max_count):
        self.max_count = max_count

    def __call__(self, y):
        counts = pd.Series(y).value_counts()
        return {c: int(min(n, self.max_count)) for c, n in counts.items()}


def make_balancer(kind="none", min_count=None, max_count=None):
    """Balanceamento/seleção de instâncias aplicado só aos dados de ajuste.

    kind: 'none' | 'class_weight' | 'ros' | 'rus' | 'smote' | 'borderline_smote' |
          'adasyn' | 'smote_tomek' | 'smote_enn' | 'tomek' | 'enn' | 'hybrid'
    'class_weight' não reamostra: é tratado em build_pipeline via o modelo.
    'hybrid' = RUS de 'normal' até max_count + SMOTE de 'anomaly' até min_count.
    """
    rs = config.RANDOM_STATE
    over = OversampleTo(min_count) if min_count else "not majority"
    under = UndersampleTo(max_count) if max_count else "auto"
    if kind in ("none", "class_weight"):
        return None
    if kind == "ros":
        return RandomOverSampler(sampling_strategy=over, random_state=rs)
    if kind == "rus":
        return RandomUnderSampler(sampling_strategy=under, random_state=rs)
    if kind == "smote":
        return SMOTE(sampling_strategy=over, random_state=rs)
    if kind == "borderline_smote":
        return BorderlineSMOTE(sampling_strategy=over, random_state=rs)
    if kind == "adasyn":
        return ADASYN(sampling_strategy=over, random_state=rs)
    if kind == "smote_tomek":
        return SMOTETomek(smote=SMOTE(sampling_strategy=over, random_state=rs), random_state=rs)
    if kind == "smote_enn":
        return SMOTEENN(smote=SMOTE(sampling_strategy=over, random_state=rs), random_state=rs)
    if kind == "tomek":
        return TomekLinks()
    if kind == "enn":
        return EditedNearestNeighbours()
    if kind == "hybrid":
        return [("rus", RandomUnderSampler(sampling_strategy=UndersampleTo(max_count or 20000),
                                           random_state=rs)),
                ("smote", SMOTE(sampling_strategy=OversampleTo(min_count or 5000),
                                random_state=rs))]
    raise ValueError(kind)

# ---------------------------------------------------------------- pipeline

def build_pipeline(model, *, redundancy="none", domain="nan", imputation="median",
                   service_encoding="onehot", categorical_encoding="onehot",
                   scaling="standard", log_transform=True, reduction="none",
                   reduction_k=20, balancing="none", min_count=None, max_count=None,
                   engineered=False):
    """Monta o Pipeline completo. Os nomes dos argumentos viram colunas do log.

    engineered=True acrescenta os atributos do FeatureBuilder (regra fixa por linha).
    """
    if imputation == "none" and (reduction != "none" or balancing not in ("none", "class_weight")):
        raise ValueError("Redução e reamostragem não aceitam NaN: escolha uma imputação.")
    model = clone(model)
    if balancing == "class_weight":
        if "class_weight" not in model.get_params():
            raise ValueError(f"{type(model).__name__} não aceita class_weight")
        model.set_params(class_weight="balanced")
    steps = []
    red = make_redundancy_sampler(redundancy)
    if red is not None:
        steps.append(("redundancy", red))
    steps.append(("domain", DomainCleaner(domain)))
    extra = ()
    if engineered:
        steps.append(("features", FeatureBuilder()))
        extra = FeatureBuilder.NAMES
    steps.append(("prep", make_preprocessor(service_encoding, categorical_encoding, scaling,
                                            log_transform, imputation, extra=extra)))
    reducer = make_reducer(reduction, k=reduction_k)
    if reducer is not None:
        steps.append(("reduce", reducer))
    bal = make_balancer(balancing, min_count, max_count)
    if isinstance(bal, list):
        steps.extend(bal)
    elif bal is not None:
        steps.append(("balance", bal))
    steps.append(("model", model))
    return Pipeline(steps)
