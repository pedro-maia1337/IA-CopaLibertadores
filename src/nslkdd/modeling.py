"""Experimentos da Etapa 3: catálogo de algoritmos, busca de hiperparâmetros e finalistas.

Compartilhado por notebooks/src/03_modelagem.py e tools/rodar_experimentos.py, para que os dois
montem exatamente os mesmos modelos e reaproveitem o cache do ExperimentLog (a chave do cache
depende dos parâmetros do modelo e do pré-processamento).

Tudo roda no treino, por CV agrupada por duplicata. Regra de escolha, declarada antes: maior
F1-macro de CV; diferença menor que 1 desvio-padrão é empate, e o empate vai para o modelo mais
simples e explicável.
"""
from sklearn.model_selection import ParameterSampler

from . import config
from .models import SoftVote, make_model
from .preprocessing import build_pipeline
from .search import FOLD_JOBS, threads

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

# Segunda rodada, só para o HGB (o mais rápido e o melhor da primeira): espaço mais largo,
# com amostragem de colunas (max_features), que também habilita a média de sementes.
SEARCH_HGB2 = {"max_iter": [500, 800, 1200, 1600], "learning_rate": [0.03, 0.05, 0.08],
               "max_leaf_nodes": [15, 31, 63, 127], "min_samples_leaf": [3, 5, 10, 20],
               "l2_regularization": [0.0, 0.1, 1.0, 5.0], "max_features": [0.5, 0.7, 1.0]}
N_SEARCH_HGB2 = 20
SEEDS = [config.RANDOM_STATE + k for k in range(5)]


def catalog(best):
    """Pré-processamento de cada algoritmo, herdado da Etapa 2 (melhores_configuracoes.json)."""
    tree, rf, mlp = best["HGB_native"], best["RF"], best["MLP"]
    return {"DT": rf, "RF": rf, "ET": rf, "HGB_native": tree, "XGB": tree, "MLP": mlp,
            "KNN": {**mlp, "reduction": "kbest_f", "reduction_k": 30},
            "NB": {**mlp, "reduction": "none"}}


def run_catalog(log, X, y, groups, best, verbose=True):
    """Cada algoritmo com os hiperparâmetros padrão do catálogo (CV agrupada 5-fold)."""
    out = {}
    for name, params in catalog(best).items():
        out[name] = log.run("etapa3", name, make_model(name, threads(name)), X, y, groups,
                            params, n_jobs=FOLD_JOBS)
        if verbose:
            r = out[name]
            print(f"{name:11s} padrão -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} "
                  f"({r['cv_seconds']:.0f}s)", flush=True)
    return out


def tuned(log, name, params, X, y, groups, space=None, n_iter=None, stage="etapa3_busca",
          verbose=True):
    """Busca aleatória: mesmo protocolo de CV e mesmo log dos demais experimentos.

    Devolve (melhores hiperparâmetros, lista de (hiperparâmetros, resultado)).
    """
    grid = ParameterSampler(space or SEARCH[name], n_iter or N_SEARCH[name],
                            random_state=config.RANDOM_STATE)
    trials = []
    for hp in grid:
        model = make_model(name, threads(name), **hp)
        r = log.run(stage, name, model, X, y, groups, params, n_jobs=FOLD_JOBS)
        trials.append((hp, r))
        if verbose:
            print(f"{name:11s} {hp} -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}",
                  flush=True)
    best_hp = max(trials, key=lambda t: t[1]["f1_macro_mean"])[0]
    return best_hp, trials


def screen_engineering(log, X, y, groups, best, hps, verbose=True):
    """Mesmos modelos ajustados, com e sem os atributos do FeatureBuilder (CV 5-fold)."""
    out = {}
    cat = catalog(best)
    for name in ("HGB_native", "XGB", "RF"):
        model = make_model(name, threads(name), **hps[name])
        for eng in (False, True):
            params = {**cat[name], "engineered": True} if eng else cat[name]
            r = log.run("etapa3_atributos", name, model, X, y, groups, params, n_jobs=FOLD_JOBS)
            out[(name, eng)] = r
            if verbose:
                print(f"{name:11s} engenharia={eng!s:5s} -> {r['f1_macro_mean']:.4f} ± "
                      f"{r['f1_macro_std']:.4f}", flush=True)
    return out


def seed_bag(hp, params, seeds=SEEDS):
    """Média das probabilidades do mesmo HGB com sementes diferentes (exige max_features < 1)."""
    members = [(f"hgb{s}", build_pipeline(make_model("HGB_native", 1, **{**hp, "random_state": s}),
                                          **params)) for s in seeds]
    return SoftVote(members)


def round2_best(log, X, y, groups, best, verbose=False):
    """Melhor configuração da busca ampliada do HGB, sem e com atributos derivados."""
    tree = catalog(best)["HGB_native"]
    out = {}
    for eng in (False, True):
        params = {**tree, "engineered": True} if eng else tree
        out[eng] = tuned(log, "HGB_native", params, X, y, groups, SEARCH_HGB2, N_SEARCH_HGB2,
                         stage="etapa3_busca2", verbose=verbose)
    return out


def new_finalists(best, hps, hgb2):
    """Candidatos que vieram das tentativas de ganho (Seção 4), para a CV repetida 3x5.

    hps: hiperparâmetros da 1ª busca; hgb2: {engineered: melhores hiperparâmetros da 2ª}.
    Devolve [(nome, pipeline pronto, parâmetros descritivos)].
    """
    tree = catalog(best)["HGB_native"]
    eng = {**tree, "engineered": True}
    desc = {"redundancy": tree["redundancy"], "repeticoes": 3}
    hgb = lambda hp, params: build_pipeline(make_model("HGB_native", 1, **hp), **params)  # noqa: E731
    bag_hp = {**hgb2[False], "max_features": 0.7}
    return [
        ("HGB + atributos", hgb(hps["HGB_native"], eng), {**desc, "engineered": True}),
        ("HGB busca 2", hgb(hgb2[False], tree), desc),
        ("HGB busca 2 + atributos", hgb(hgb2[True], eng), {**desc, "engineered": True}),
        ("HGB busca 2, 5 sementes", seed_bag(bag_hp, tree), {**desc, "sementes": len(SEEDS)}),
        ("Voto HGB busca 2 ± atributos",
         SoftVote([("hgb", hgb(hgb2[False], tree)), ("hgb_eng", hgb(hgb2[True], eng))]),
         {**desc, "engineered": "membro"}),
    ]


def run_finalists(log, X, y, groups, candidates, verbose=True):
    """CV agrupada 5-fold repetida 3 vezes (15 folds) para os candidatos prontos."""
    out = {}
    for name, pipe, params in candidates:
        out[name] = log.run("etapa3_final", name, pipe, X, y, groups, params, n_repeats=3,
                            prebuilt=True, n_jobs=FOLD_JOBS)
        if verbose:
            r = out[name]
            print(f"{name:30s} 3x5 -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} "
                  f"limiar {r['threshold_mean']:.2f} ({r['cv_seconds']:.0f}s)", flush=True)
    return out


# ------------------------------------------------- algoritmos fora da disciplina (extra)
# LightGBM e CatBoost: boosting de árvores com categorias nativas, referência em dados
# tabulares. Usam o mesmo pré-processamento do HGB (limpeza de rótulos, NaN nativo).
EXTERNAL = ("LGBM", "CatBoost")
SEARCH_EXT = {
    "LGBM": {"n_estimators": [300, 600, 1000, 1500], "learning_rate": [0.02, 0.05, 0.08],
             "num_leaves": [15, 31, 63, 127], "min_child_samples": [5, 10, 20, 40],
             "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.5, 0.7, 1.0],
             "reg_lambda": [0.0, 0.1, 1.0, 5.0]},
    "CatBoost": {"iterations": [500, 1000, 1500], "learning_rate": [0.03, 0.05, 0.08, 0.12],
                 "depth": [4, 6, 8], "l2_leaf_reg": [1, 3, 10]},
}
N_SEARCH_EXT = {"LGBM": 20, "CatBoost": 8}


def run_external(log, X, y, groups, best, verbose=True):
    """Padrão e busca aleatória de LightGBM e CatBoost (CV agrupada 5-fold).

    Devolve ({nome: resultado padrão}, {nome: (melhores hiperparâmetros, tentativas)}).
    """
    tree = catalog(best)["HGB_native"]
    defaults, searches = {}, {}
    for name in EXTERNAL:
        defaults[name] = log.run("etapa3_externos", name, make_model(name, threads(name)), X, y,
                                 groups, tree, n_jobs=FOLD_JOBS)
        if verbose:
            r = defaults[name]
            print(f"{name:9s} padrão -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} "
                  f"({r['cv_seconds']:.0f}s)", flush=True)
        searches[name] = tuned(log, name, tree, X, y, groups, SEARCH_EXT[name],
                               N_SEARCH_EXT[name], stage="etapa3_externos_busca", verbose=verbose)
    return defaults, searches


def external_finalists(best, hgb2, ext_hps):
    """LightGBM e CatBoost ajustados e votos com o HGB da busca ampliada, para a CV 3x5."""
    tree = catalog(best)["HGB_native"]
    desc = {"redundancy": tree["redundancy"], "repeticoes": 3}
    pipe = lambda n, hp: build_pipeline(make_model(n, threads(n), **hp), **tree)  # noqa: E731
    hgb = build_pipeline(make_model("HGB_native", 1, **hgb2[False]), **tree)
    lgbm, cb = pipe("LGBM", ext_hps["LGBM"]), pipe("CatBoost", ext_hps["CatBoost"])
    return [
        ("LightGBM", lgbm, desc),
        ("CatBoost", cb, desc),
        ("Voto HGB + LightGBM", SoftVote([("hgb", hgb), ("lgbm", lgbm)]), desc),
        ("Voto HGB + CatBoost", SoftVote([("hgb", hgb), ("catboost", cb)]), desc),
        ("Voto HGB + LightGBM + CatBoost",
         SoftVote([("hgb", hgb), ("lgbm", lgbm), ("catboost", cb)]), desc),
    ]
