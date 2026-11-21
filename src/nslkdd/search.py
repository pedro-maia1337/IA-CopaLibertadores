"""Busca por coordenadas dos eixos de pré-processamento (Etapa 2).

Compartilhado por notebooks/src/02_preprocessamento.py e tools/rodar_experimentos.py, para que
os dois chamem exatamente as mesmas configurações e reaproveitem o cache do ExperimentLog.

Regra de decisão pré-declarada: a primeira opção de cada eixo é a mais simples; uma alternativa
só a substitui se superar o F1-macro dela em mais de 1 desvio-padrão (entre folds).
"""
import pandas as pd

from .models import make_model

FOLD_JOBS = 5

TREE_START = {"redundancy": "none", "domain": "nan", "imputation": "none", "scaling": "none",
              "log_transform": False, "service_encoding": "ordinal",
              "categorical_encoding": "ordinal"}
# Ponto de partida de cada família: a configuração mais simples que o algoritmo aceita.
FAMILIES = {
    "HGB_native": dict(TREE_START),
    "RF": {**TREE_START, "imputation": "median", "service_encoding": "onehot",
           "categorical_encoding": "onehot"},
    "MLP": {"redundancy": "none", "domain": "nan", "imputation": "median", "scaling": "standard",
            "log_transform": True, "service_encoding": "onehot"},
}
REDUNDANCY = [{"redundancy": v} for v in
              ["none", "drop_duplicates", "drop_inconsistent", "majority_label", "relabel_model"]]
DOMAIN = [{"domain": v} for v in ["nan", "none", "clip"]]
SERVICE = [{"service_encoding": v} for v in ["onehot", "onehot_rare", "frequency", "target"]]
BALANCE = [{"balancing": "none"}, {"balancing": "class_weight"},
           {"balancing": "ros", "min_count": 10000}, {"balancing": "rus", "max_count": 10000},
           {"balancing": "smote", "min_count": 10000},
           {"balancing": "borderline_smote", "min_count": 10000},
           {"balancing": "smote_tomek", "min_count": 10000},
           {"balancing": "hybrid", "min_count": 5000, "max_count": 20000}]
REDUCE = [{"reduction": "none"}, {"reduction": "pca"}, {"reduction": "kbest_f", "reduction_k": 30},
          {"reduction": "kbest_mi", "reduction_k": 30}, {"reduction": "tree_select"}]
SCALE = [{"scaling": "standard", "log_transform": True},
         {"scaling": "standard", "log_transform": False},
         {"scaling": "minmax", "log_transform": True},
         {"scaling": "robust", "log_transform": True},
         {"scaling": "quantile", "log_transform": False}]
AXES = {
    "HGB_native": [
        ("redundancia", REDUNDANCY), ("dominio", DOMAIN),
        ("imputacao", [{"imputation": "none"}, {"imputation": "median"},
                       {"imputation": "median_indicator"}]),
        ("balanceamento", [o for o in BALANCE if o["balancing"] in ("none", "class_weight")]),
    ],
    "RF": [
        ("redundancia", REDUNDANCY),
        ("imputacao", [{"imputation": "median"}, {"imputation": "none"},
                       {"imputation": "median_indicator"}, {"imputation": "mean"}]),
        ("codificacao", SERVICE + [{"service_encoding": "ordinal"}]),
        ("balanceamento", BALANCE), ("reducao", REDUCE),
    ],
    "MLP": [
        ("redundancia", REDUNDANCY[:3]), ("dominio", DOMAIN),
        ("imputacao", [{"imputation": "median"}, {"imputation": "mean"},
                       {"imputation": "median_indicator"}, {"imputation": "iterative"}]),
        ("codificacao", SERVICE),
        ("escala", SCALE),
        ("balanceamento", [o for o in BALANCE if o["balancing"] != "class_weight"]),
        ("reducao", REDUCE),
    ],
}


def threads(name):
    """Threads por modelo; com FOLD_JOBS folds em paralelo, cabe em 16 núcleos."""
    return 3 if name in ("RF", "ET", "XGB", "KNN", "LGBM", "CatBoost") else 1


def pick(results, options):
    """Mantém a primeira opção salvo ganho maior que 1 dp sobre ela."""
    reference = results[0]
    best, chosen = reference, options[0]
    for result, option in zip(results[1:], options[1:]):
        if (result["f1_macro_mean"] > best["f1_macro_mean"]
                and result["f1_macro_mean"] - reference["f1_macro_mean"] > reference["f1_macro_std"]):
            best, chosen = result, option
    return chosen, best


def describe(option):
    """Rótulo curto de uma opção, para tabelas e gráficos."""
    return ", ".join(f"{v}" if k in ("redundancy", "domain", "imputation", "service_encoding",
                                     "balancing", "reduction", "scaling")
                     else f"{k}={v}" for k, v in option.items())


def run_axis(log, family, axis, options, params, X, y, groups, verbose=True):
    """Roda (ou lê do cache) todas as opções de um eixo, com o resto fixo em `params`."""
    rows, results = [], []
    for option in options:
        model = make_model(family, threads(family))
        r = log.run(f"etapa2_{axis}", family, model, X, y, groups, {**params, **option},
                    n_jobs=FOLD_JOBS)
        results.append(r)
        rows.append({"familia": family, "eixo": axis, "opcao": describe(option),
                     "f1_macro": r["f1_macro_mean"], "dp": r["f1_macro_std"],
                     "f1_macro_bruto": r["f1_macro_noisy_mean"],
                     "f1_normal": r["f1_normal_mean"], "f1_anomaly": r["f1_anomaly_mean"],
                     "precision_anomaly": r["precision_anomaly_mean"],
                     "recall_anomaly": r["recall_anomaly_mean"],
                     "limiar": r["threshold_mean"], "segundos": r["cv_seconds"]})
        if verbose:
            print(f"{family:11s} {axis:13s} {describe(option):32s} -> {r['f1_macro_mean']:.4f} "
                  f"± {r['f1_macro_std']:.4f}", flush=True)
    chosen, _ = pick(results, options)
    table = pd.DataFrame(rows)
    table["escolhida"] = table["opcao"] == describe(chosen)
    table["ganho_vs_ref"] = (table["f1_macro"] - table["f1_macro"].iloc[0]).round(4)
    return chosen, table


def coordinate_search(log, X, y, groups, families=None, verbose=True):
    """Busca um eixo por vez para cada família. Devolve (melhores parâmetros, tabela longa)."""
    best, tables = {}, []
    for family in families or FAMILIES:
        params = dict(FAMILIES[family])
        for axis, options in AXES[family]:
            chosen, table = run_axis(log, family, axis, options, params, X, y, groups, verbose)
            params.update(chosen)
            tables.append(table)
        best[family] = params
    return best, pd.concat(tables, ignore_index=True)
