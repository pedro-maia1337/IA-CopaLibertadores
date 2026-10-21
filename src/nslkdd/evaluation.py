"""Validação cruzada no treino, registro de experimentos e avaliação final."""
import hashlib
import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score,
                             classification_report, f1_score, make_scorer)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate

from . import config


def make_cv(kind="stratified", n_splits=5, attack=None):
    """Esquemas de validação, sempre só sobre o KDDTrain+.

    'stratified'   : StratifiedKFold padrão (métrica principal de seleção).
    'attack_group' : cada tipo de ataque fica inteiro num único fold, simulando
                     os ataques inéditos do KDDTest+ (diagnóstico de generalização).
                     Devolve (cv, groups); passe groups para cross_validate.
    """
    if kind == "stratified":
        return StratifiedKFold(n_splits, shuffle=True, random_state=config.RANDOM_STATE), None
    if kind == "attack_group":
        if attack is None:
            raise ValueError("attack_group exige a série 'attack' do treino")
        rng = np.random.default_rng(config.RANDOM_STATE)
        groups = attack.to_numpy(dtype=object).copy()
        is_normal = groups == "normal"
        groups[is_normal] = [f"normal_{i}" for i in rng.integers(0, 50, is_normal.sum())]
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        return cv, groups
    raise ValueError(kind)


def scorers(classes=config.CLASSES):
    s = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy",
         "accuracy": "accuracy"}
    for c in classes:
        s[f"f1_{c}"] = make_scorer(f1_score, labels=[c], average="macro", zero_division=0)
    return s


def cross_validate_pipeline(pipe, X, y, cv, groups=None, n_jobs=-1):
    """Roda CV e devolve médias/desvios de F1-macro, acurácia balanceada e F1 por classe."""
    t0 = time.time()
    res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scorers(),
                         n_jobs=n_jobs, error_score="raise")
    out = {"cv_seconds": round(time.time() - t0, 1)}
    for k, v in res.items():
        if k.startswith("test_"):
            name = k[5:]
            out[f"{name}_mean"] = round(float(np.mean(v)), 4)
            if name in ("f1_macro", "balanced_accuracy"):
                out[f"{name}_std"] = round(float(np.std(v)), 4)
    return out


class ExperimentLog:
    """Tabela comparativa persistente (results/experiments.csv).

    Cada linha = configuração de pré-processamento x algoritmo x métricas de CV.
    Configurações já executadas são puladas (cache por hash), o que permite
    reexecutar notebooks sem refazer horas de CV.
    """

    def __init__(self, path=config.EXPERIMENT_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def table(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    @staticmethod
    def key(stage, model_name, params, cv_kind):
        blob = json.dumps([stage, model_name, params, cv_kind], sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def run(self, stage, model_name, model, X, y, params=None, cv_kind="stratified",
            n_splits=5, attack=None, force=False, n_jobs=-1):
        from .preprocessing import build_pipeline

        params = params or {}
        key = self.key(stage, model_name, params, cv_kind)
        table = self.table()
        if not force and not table.empty and key in set(table["key"]):
            return table[table["key"] == key].iloc[-1].to_dict()
        cv, groups = make_cv(cv_kind, n_splits, attack)
        pipe = build_pipeline(model, **params)
        row = {"key": key, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "stage": stage, "model": model_name, "cv": f"{cv_kind}-{n_splits}",
               **{f"p_{k}": v for k, v in params.items()},
               "model_params": json.dumps(model.get_params(), default=str)[:500],
               **cross_validate_pipeline(pipe, X, y, cv, groups, n_jobs)}
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table.to_csv(self.path, index=False)
        return row

    def leaderboard(self, stage=None, cols=None):
        t = self.table()
        if stage is not None:
            t = t[t["stage"] == stage]
        cols = cols or [c for c in t.columns if c.startswith("p_")] + [
            "model", "f1_macro_mean", "f1_macro_std", "balanced_accuracy_mean",
            *[f"f1_{c}_mean" for c in config.CLASSES], "cv_seconds"]
        return t[[c for c in cols if c in t.columns]].sort_values("f1_macro_mean", ascending=False)


def final_report(y_true, y_pred, title="KDDTest+", save_as=None):
    """Métricas oficiais do teste: F1-macro + relatório por classe + matriz de confusão."""
    labels = config.CLASSES
    print(f"F1-macro ({title}): {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Acurácia balanceada: {balanced_accuracy_score(y_true, y_pred):.4f}\n")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, norm in zip(axes, [None, "true"]):
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize=norm,
                                                values_format=".2f" if norm else "d",
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{title} — {'recall por classe' if norm else 'contagens'}")
    fig.tight_layout()
    if save_as:
        fig.savefig(config.FIGURES / save_as, dpi=150, bbox_inches="tight")
    return fig
