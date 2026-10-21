"""Blocos de pré-processamento e montagem de pipelines imblearn.

Tudo que aprende parâmetros (codificação, escala, seleção, balanceamento,
remoção de redundâncias) fica DENTRO do Pipeline, então em validação cruzada é
ajustado só nos folds de treino e nunca enxerga o fold de validação ou o teste.
Ordem: redundância -> codificação/transformação -> redução -> balanceamento -> modelo.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, QuantileTransformer, RobustScaler,
                                   StandardScaler, TargetEncoder)

from . import config

# ---------------------------------------------------------------- redundância

def _resolve_redundancy(X, y, strategy):
    df = X.copy()
    df["__y"] = np.asarray(y)
    feats = list(X.columns)
    if strategy == "drop_duplicates":
        df = df.drop_duplicates()
    elif strategy == "drop_inconsistent":
        n_labels = df.groupby(feats, dropna=False)["__y"].transform("nunique")
        df = df[n_labels == 1].drop_duplicates()
    elif strategy == "majority_label":
        mode = df.groupby(feats, dropna=False)["__y"].transform(lambda s: s.mode().iat[0])
        df = df.assign(__y=mode).drop_duplicates()
    else:
        raise ValueError(strategy)
    return df[feats], df["__y"].to_numpy()


def make_redundancy_sampler(strategy: str = "none"):
    """'none' | 'drop_duplicates' | 'drop_inconsistent' | 'majority_label'."""
    if strategy == "none":
        return None
    return FunctionSampler(func=_resolve_redundancy, validate=False,
                           kw_args={"strategy": strategy})

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
        return np.column_stack([X[c].map(m).fillna(0.0).to_numpy(float)
                                for c, m in self.maps_.items()])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


def make_service_encoder(kind: str):
    """Codificação do atributo de alta cardinalidade 'service' (70 valores)."""
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "onehot_rare":  # agrupa serviços com < 0,1% das linhas em 'infrequent'
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                             sparse_output=False)
    if kind == "frequency":
        return FrequencyEncoder()
    if kind == "target":  # cross-fitting interno evita vazamento do rótulo
        return TargetEncoder(target_type="multiclass",
                             cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE))
    if kind == "ordinal":  # só faz sentido para árvores
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(kind)


def make_scaler(kind: str):
    return {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "quantile": lambda: QuantileTransformer(output_distribution="normal",
                                                random_state=config.RANDOM_STATE),
        "none": lambda: "passthrough",
    }[kind]()


def make_preprocessor(service_encoding="onehot", scaling="standard", log_transform=True,
                      drop=tuple(config.CONSTANT_IN_TRAIN)):
    numeric = [c for c in config.NUMERIC if c not in drop]
    heavy = [c for c in config.HEAVY_TAILED if c in numeric] if log_transform else []
    light = [c for c in numeric if c not in heavy]
    scaler = make_scaler(scaling)
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["protocol_type", "flag"]),
        ("service", make_service_encoder(service_encoding), ["service"]),
        ("num", clone(scaler) if scaler != "passthrough" else scaler, light),
    ]
    if heavy:
        log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        transformers.append(("log", make_pipeline(log, clone(scaler)) if scaler != "passthrough"
                             else log, heavy))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

# ---------------------------------------------------------------- redução

def make_reducer(kind="none", k=20, variance=0.95):
    """'none' | 'pca' | 'kbest_f' | 'kbest_mi' | 'tree_select'."""
    if kind == "none":
        return None
    if kind == "pca":
        return PCA(n_components=variance, random_state=config.RANDOM_STATE)
    if kind == "kbest_f":
        return SelectKBest(f_classif, k=k)
    if kind == "kbest_mi":
        return SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=config.RANDOM_STATE), k=k)
    if kind == "tree_select":
        return SelectFromModel(ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                                                    random_state=config.RANDOM_STATE),
                               threshold="median")
    raise ValueError(kind)

# ---------------------------------------------------------------- balanceamento

def oversample_to(min_count):
    """sampling_strategy: eleva cada classe a pelo menos min_count instâncias."""
    def strategy(y):
        counts = pd.Series(y).value_counts()
        return {c: int(max(n, min_count)) for c, n in counts.items()}
    return strategy


def undersample_to(max_count):
    """sampling_strategy: reduz cada classe a no máximo max_count instâncias."""
    def strategy(y):
        counts = pd.Series(y).value_counts()
        return {c: int(min(n, max_count)) for c, n in counts.items()}
    return strategy


def make_balancer(kind="none", min_count=None, max_count=None):
    """Balanceamento/seleção de instâncias aplicado só aos dados de ajuste.

    kind: 'none' | 'class_weight' | 'ros' | 'rus' | 'smote' | 'borderline_smote' |
          'adasyn' | 'smote_tomek' | 'smote_enn' | 'tomek' | 'enn' | 'hybrid'
    'class_weight' não reamostra: é tratado em build_pipeline via o modelo.
    'hybrid' = RUS das classes grandes até max_count + SMOTE das pequenas até min_count.
    """
    rs = config.RANDOM_STATE
    over = oversample_to(min_count) if min_count else "not majority"
    under = undersample_to(max_count) if max_count else "auto"
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
        return TomekLinks(n_jobs=-1)
    if kind == "enn":
        return EditedNearestNeighbours(n_jobs=-1)
    if kind == "hybrid":
        return [("rus", RandomUnderSampler(sampling_strategy=undersample_to(max_count or 20000),
                                           random_state=rs)),
                ("smote", SMOTE(sampling_strategy=oversample_to(min_count or 2000),
                                random_state=rs))]
    raise ValueError(kind)

# ---------------------------------------------------------------- pipeline

def build_pipeline(model, *, redundancy="none", service_encoding="onehot",
                   scaling="standard", log_transform=True, reduction="none",
                   reduction_k=20, balancing="none", min_count=None, max_count=None):
    """Monta o Pipeline completo. Os nomes dos argumentos viram colunas do log."""
    model = clone(model)
    if balancing == "class_weight":
        if "class_weight" not in model.get_params():
            raise ValueError(f"{type(model).__name__} não aceita class_weight")
        model.set_params(class_weight="balanced")
    steps = []
    red = make_redundancy_sampler(redundancy)
    if red is not None:
        steps.append(("redundancy", red))
    steps.append(("prep", make_preprocessor(service_encoding, scaling, log_transform)))
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
