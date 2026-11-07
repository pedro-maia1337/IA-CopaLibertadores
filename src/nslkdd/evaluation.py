"""Validação cruzada no treino, registro de experimentos e relatório de validação."""
import hashlib
import json
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from . import config
from .data import evaluation_mask
from .decision import best_threshold, cross_val_proba

# A professora divulga precisão, recall e F1 por classe (enunciado, Seção 3.1); o ranking é por F1-macro.
METRICS = ["f1_macro", "f1_normal", "f1_anomaly", "precision_normal", "recall_normal",
           "precision_anomaly", "recall_anomaly", "balanced_accuracy", "accuracy"]


def binary_metrics(is_positive, predicted_positive):
    """Métricas do problema binário a partir de máscaras booleanas (anomaly = positivo)."""
    tp = np.sum(is_positive & predicted_positive)
    fp = np.sum(~is_positive & predicted_positive)
    fn = np.sum(is_positive & ~predicted_positive)
    tn = np.sum(~is_positive & ~predicted_positive)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fp + fn, 1)
    recall, specificity = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"f1_macro": (f1_pos + f1_neg) / 2, "f1_normal": f1_neg, "f1_anomaly": f1_pos,
            "precision_normal": tn / max(tn + fn, 1), "recall_normal": specificity,
            "precision_anomaly": tp / max(tp + fp, 1), "recall_anomaly": recall,
            "balanced_accuracy": (recall + specificity) / 2,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1)}


def _score_folds(proba, fold, y, mask):
    """Cada fold usa um limiar aprendido nos demais: o limiar nunca vê o fold que pontua."""
    rows = []
    for k in np.unique(fold):
        own = fold == k
        rest = mask & ~own
        threshold = best_threshold(proba[rest], y[rest])
        score = mask & own
        row = binary_metrics(y[score], proba[score] >= threshold)
        row["threshold"] = threshold
        row["f1_macro_05"] = binary_metrics(y[score], proba[score] >= 0.5)["f1_macro"]
        row["f1_macro_noisy"] = binary_metrics(y[own], proba[own] >= threshold)["f1_macro"]
        rows.append(row)
    return rows


def cross_validate_pipeline(pipe, X, y, groups, n_splits=5, n_repeats=1, n_jobs=-1):
    """CV por grupo de duplicatas, com limiar cruzado.

    Métrica principal (f1_macro): linhas únicas e sem rótulo contraditório. f1_macro_noisy
    inclui cópias e conflitos e serve de contraste; o teto dela é baixo por construção.
    """
    t0 = time.time()
    is_positive = y.to_numpy() == config.POSITIVE
    mask = evaluation_mask(groups, y)
    rows = []
    for r in range(n_repeats):
        proba, fold = cross_val_proba(pipe, X, y, groups, n_splits,
                                      seed=config.RANDOM_STATE + r, n_jobs=n_jobs)
        rows += _score_folds(proba, fold, is_positive, mask)
    table = pd.DataFrame(rows)
    out = {"cv_seconds": round(time.time() - t0, 1)}
    for name in table.columns:
        out[f"{name}_mean"] = round(float(table[name].mean()), 4)
    out["f1_macro_std"] = round(float(table["f1_macro"].std(ddof=0)), 4)
    return out


def _data_version():
    return hashlib.md5(config.TRAIN_FILE.read_bytes()).hexdigest()[:8]


class ExperimentLog:
    """Tabela comparativa persistente (results/experiments.csv).

    Cada linha = configuração de pré-processamento x algoritmo x métricas de CV.
    Configurações já executadas são puladas (cache por hash), o que permite
    reexecutar notebooks sem refazer horas de CV.
    """

    def __init__(self, path=config.EXPERIMENT_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data_version = _data_version()

    def table(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def key(self, stage, model_name, params, model, n_splits, n_repeats):
        blob = json.dumps([stage, model_name, params, model.get_params(), n_splits, n_repeats,
                           self.data_version], sort_keys=True, default=str)
        blob = re.sub(r"0x[0-9a-fA-F]+", "", blob)  # endereços de memória mudam a cada execução
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def run(self, stage, model_name, model, X, y, groups, params=None, n_splits=5, n_repeats=1,
            prebuilt=False, force=False, n_jobs=-1):
        """prebuilt=True: 'model' já é o pipeline/ensemble pronto, e 'params' só descreve a linha."""
        from .preprocessing import build_pipeline

        params = params or {}
        key = self.key(stage, model_name, params, model, n_splits, n_repeats)
        table = self.table()
        if not force and not table.empty and key in set(table["key"]):
            return table[table["key"] == key].iloc[-1].to_dict()
        pipe = model if prebuilt else build_pipeline(model, **params)
        row = {"key": key, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "stage": stage, "model": model_name, "cv": f"group-{n_splits}x{n_repeats}",
               "data_version": self.data_version,
               **{f"p_{k}": v for k, v in params.items()},
               "model_params": json.dumps(model.get_params(), default=str)[:500],
               **cross_validate_pipeline(pipe, X, y, groups, n_splits, n_repeats, n_jobs)}
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table.to_csv(self.path, index=False)
        return row

    def leaderboard(self, stage=None, cols=None):
        t = self.table()
        if stage is not None:
            t = t[t["stage"] == stage]
        cols = cols or [c for c in t.columns if c.startswith("p_")] + [
            "model", "f1_macro_mean", "f1_macro_std", "f1_macro_05_mean", "f1_macro_noisy_mean",
            "f1_normal_mean", "f1_anomaly_mean", "precision_normal_mean", "recall_normal_mean",
            "precision_anomaly_mean", "recall_anomaly_mean",
            "balanced_accuracy_mean", "threshold_mean", "cv_seconds"]
        return t[[c for c in cols if c in t.columns]].sort_values("f1_macro_mean", ascending=False)


def validation_report(y_true, y_pred, title="OOF no treino", save_as=None):
    """F1-macro, relatório por classe e matriz de confusão. O teste não tem rótulos aqui."""
    is_positive = np.asarray(y_true) == config.POSITIVE
    metrics = binary_metrics(is_positive, np.asarray(y_pred) == config.POSITIVE)
    labels = config.CLASSES
    print(f"F1-macro ({title}): {metrics['f1_macro']:.4f}")
    print(f"Acurácia balanceada: {metrics['balanced_accuracy']:.4f}\n")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, norm in zip(axes, [None, "true"]):
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize=norm,
                                                values_format=".3f" if norm else "d",
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{title} — {'recall por classe' if norm else 'contagens'}")
    fig.tight_layout()
    if save_as:
        fig.savefig(config.FIGURES / save_as, dpi=150, bbox_inches="tight")
    return fig
