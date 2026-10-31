"""Busca por coordenadas dos eixos de pré-processamento e comparação de algoritmos.

Tudo roda no treino, por CV agrupada, e fica em results/experiments.csv (com cache: reexecutar
só calcula o que falta). Regra de decisão pré-declarada: uma opção só substitui a anterior
se superar a mais simples em mais de 1 desvio-padrão; diferença menor não é diferença.

Uso:  python tools/rodar_experimentos.py etapa2 | etapa3
Os fatos usados pelas etapas 2 e 3 ficam em results/melhores_configuracoes.json.
"""
import argparse
import json
import warnings

from sklearn.model_selection import ParameterSampler

from nslkdd import ExperimentLog, build_pipeline, config, load_train
from nslkdd.models import SoftVote, make_model

warnings.filterwarnings("ignore")
BEST_FILE = config.RESULTS / "melhores_configuracoes.json"
FOLD_JOBS = 5

TREE_START = {"redundancy": "none", "domain": "nan", "imputation": "none", "scaling": "none",
              "log_transform": False, "service_encoding": "ordinal",
              "categorical_encoding": "ordinal"}
FAMILIES = {
    "HGB_native": dict(TREE_START),
    "RF": {**TREE_START, "imputation": "median", "service_encoding": "onehot",
           "categorical_encoding": "onehot"},
    "MLP": {"redundancy": "none", "domain": "nan", "imputation": "median", "scaling": "standard",
            "log_transform": True, "service_encoding": "onehot"},
}
# Primeira opção de cada eixo = a mais simples (referência da regra de decisão).
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
        ("escala", [{"scaling": "standard", "log_transform": True},
                    {"scaling": "standard", "log_transform": False},
                    {"scaling": "minmax", "log_transform": True},
                    {"scaling": "robust", "log_transform": True},
                    {"scaling": "quantile", "log_transform": False}]),
        ("balanceamento", [o for o in BALANCE if o["balancing"] != "class_weight"]),
        ("reducao", REDUCE),
    ],
}
SEARCH = {
    "HGB_native": {"max_iter": [200, 300, 500, 800], "learning_rate": [0.02, 0.04, 0.06, 0.1],
                   "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [5, 10, 20, 40],
                   "l2_regularization": [0.0, 0.1, 1.0, 5.0]},
    "XGB": {"n_estimators": [300, 600, 1000], "learning_rate": [0.02, 0.05, 0.1],
            "max_depth": [4, 6, 7, 9], "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0], "min_child_weight": [1, 3, 5]},
    "RF": {"n_estimators": [300, 500], "max_features": ["sqrt", 0.3, 0.5],
           "min_samples_leaf": [1, 2, 4], "class_weight": [None, "balanced_subsample"]},
    "MLP": {"hidden_layer_sizes": [(64,), (128, 64), (256, 128)], "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [5e-4, 1e-3]},
}
N_SEARCH = {"HGB_native": 12, "XGB": 12, "RF": 6, "MLP": 6}


def threads(name):
    return 3 if name in ("RF", "ET", "XGB", "KNN") else 1


def pick(results, options):
    """Mantém a primeira opção salvo ganho maior que 1 dp sobre ela."""
    reference = results[0]
    best, chosen = reference, options[0]
    for result, option in zip(results[1:], options[1:]):
        if (result["f1_macro_mean"] > best["f1_macro_mean"]
                and result["f1_macro_mean"] - reference["f1_macro_mean"] > reference["f1_macro_std"]):
            best, chosen = result, option
    return chosen, best


def etapa2(log, X, y, groups):
    best = {}
    for family, start in FAMILIES.items():
        params = dict(start)
        for axis, options in AXES[family]:
            results = []
            for option in options:
                model = make_model(family, threads(family))
                r = log.run(f"etapa2_{axis}", family, model, X, y, groups, {**params, **option},
                            n_jobs=FOLD_JOBS)
                print(f"{family:11s} {axis:13s} {option} -> {r['f1_macro_mean']:.4f} "
                      f"± {r['f1_macro_std']:.4f} (bruto {r['f1_macro_noisy_mean']:.4f}, "
                      f"{r['cv_seconds']:.0f}s)", flush=True)
                results.append(r)
            chosen, top = pick(results, options)
            params.update(chosen)
            print(f"  => {axis}: {chosen} ({top['f1_macro_mean']:.4f})", flush=True)
        best[family] = params
    BEST_FILE.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def tuned(log, name, params, X, y, groups):
    """Busca aleatória curta: mesmo protocolo de CV e mesmo log dos demais experimentos."""
    best, best_hp = None, {}
    grid = ParameterSampler(SEARCH[name], N_SEARCH[name], random_state=config.RANDOM_STATE)
    for hp in grid:
        model = make_model(name, threads(name), **hp)
        r = log.run("etapa3_busca", name, model, X, y, groups, params, n_jobs=FOLD_JOBS)
        print(f"{name:11s} {hp} -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}", flush=True)
        if best is None or r["f1_macro_mean"] > best["f1_macro_mean"]:
            best, best_hp = r, hp
    return best_hp


def etapa3(log, X, y, groups):
    best = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    tree_params = best["HGB_native"]
    clean = tree_params["redundancy"]
    rf, mlp = best["RF"], best["MLP"]
    catalog = {"DT": rf, "RF": rf, "ET": rf, "HGB_native": tree_params, "XGB": tree_params,
               "MLP": mlp, "KNN": {**mlp, "reduction": "kbest_f", "reduction_k": 30},
               "NB": {**mlp, "reduction": "none"}}
    for name, params in catalog.items():
        r = log.run("etapa3", name, make_model(name, threads(name)), X, y, groups, params,
                    n_jobs=FOLD_JOBS)
        print(f"{name:11s} padrão -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} "
              f"({r['cv_seconds']:.0f}s)", flush=True)
    hps = {n: tuned(log, n, catalog[n], X, y, groups) for n in SEARCH}
    best["hiperparametros"] = hps
    BEST_FILE.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    finalists = {n: build_pipeline(make_model(n, threads(n), **hps[n]), **catalog[n]) for n in hps}
    voting = SoftVote([(n.lower(), finalists[n]) for n in ("HGB_native", "XGB", "RF")])
    for name, pipe in [*finalists.items(), ("Voto HGB+XGB+RF", voting)]:
        r = log.run("etapa3_final", name, pipe, X, y, groups,
                    {"redundancy": clean, "repeticoes": 3}, n_repeats=3, prebuilt=True,
                    n_jobs=FOLD_JOBS)
        print(f"{name:16s} 3x5 -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} "
              f"limiar {r['threshold_mean']:.2f} ({r['cv_seconds']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["etapa2", "etapa3"])
    args = ap.parse_args()
    log = ExperimentLog()
    X, y, groups = load_train()
    {"etapa2": etapa2, "etapa3": etapa3}[args.etapa](log, X, y, groups)


if __name__ == "__main__":
    main()
