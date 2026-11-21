"""Busca por coordenadas dos eixos de pré-processamento e comparação de algoritmos.

Tudo roda no treino, por CV agrupada, e fica em results/experiments.csv (com cache: reexecutar
só calcula o que falta). Regra de decisão pré-declarada: uma opção só substitui a anterior
se superar a mais simples em mais de 1 desvio-padrão; diferença menor não é diferença.

Uso:  python tools/rodar_experimentos.py etapa2 | etapa3 | etapa3_extra | etapa3_externos
Os fatos usados pelas etapas 2 e 3 ficam em results/melhores_configuracoes.json.
"""
import argparse
import json
import warnings

from nslkdd import ExperimentLog, build_pipeline, config, load_train
from nslkdd.modeling import (SEARCH, catalog, external_finalists, new_finalists, round2_best,
                             run_catalog, run_external, run_finalists, screen_engineering, tuned)
from nslkdd.models import SoftVote, make_model
from nslkdd.search import FOLD_JOBS, coordinate_search, threads

warnings.filterwarnings("ignore")
BEST_FILE = config.RESULTS / "melhores_configuracoes.json"


def etapa2(log, X, y, groups):
    best, _ = coordinate_search(log, X, y, groups)
    BEST_FILE.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def etapa3(log, X, y, groups):
    best = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    cat = catalog(best)
    clean = cat["HGB_native"]["redundancy"]
    run_catalog(log, X, y, groups, best)
    hps = {n: tuned(log, n, cat[n], X, y, groups)[0] for n in SEARCH}
    best["hiperparametros"] = hps
    BEST_FILE.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    finalists = {n: build_pipeline(make_model(n, threads(n), **hps[n]), **cat[n]) for n in hps}
    voting = SoftVote([(n.lower(), finalists[n]) for n in ("HGB_native", "XGB", "RF")])
    for name, pipe in [*finalists.items(), ("Voto HGB+XGB+RF", voting)]:
        r = log.run("etapa3_final", name, pipe, X, y, groups,
                    {"redundancy": clean, "repeticoes": 3}, n_repeats=3, prebuilt=True,
                    n_jobs=FOLD_JOBS)
        print(f"{name:16s} 3x5 -> {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} "
              f"limiar {r['threshold_mean']:.2f} ({r['cv_seconds']:.0f}s)", flush=True)


def etapa3_extra(log, X, y, groups):
    """Tentativas de subir o F1-macro: atributos derivados e busca mais larga no HGB."""
    best = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    cat, hps = catalog(best), best["hiperparametros"]
    screen_engineering(log, X, y, groups, best, hps)
    hgb2 = {eng: hp for eng, (hp, _) in round2_best(log, X, y, groups, best, verbose=True).items()}
    run_finalists(log, X, y, groups, new_finalists(best, hps, hgb2))


def etapa3_externos(log, X, y, groups):
    """Algoritmos fora da disciplina (LightGBM, CatBoost) e votos com o HGB."""
    best = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    hgb2 = {eng: hp for eng, (hp, _) in round2_best(log, X, y, groups, best).items()}
    _, searches = run_external(log, X, y, groups, best)
    ext_hps = {n: hp for n, (hp, _) in searches.items()}
    run_finalists(log, X, y, groups, external_finalists(best, hgb2, ext_hps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["etapa2", "etapa3", "etapa3_extra", "etapa3_externos"])
    args = ap.parse_args()
    log = ExperimentLog()
    X, y, groups = load_train()
    {"etapa2": etapa2, "etapa3": etapa3, "etapa3_extra": etapa3_extra,
     "etapa3_externos": etapa3_externos}[args.etapa](log, X, y, groups)


if __name__ == "__main__":
    main()
