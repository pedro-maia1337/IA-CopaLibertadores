"""Auditoria de vazamento de dados.

1) association_with_target: força de associação de cada atributo com a classe
   (Seção 7 do enunciado), calculada SÓ no treino.
2) scan_notebook / CLI: varredura estática de notebooks e scripts em busca de
   padrões clássicos de vazamento.

Uso:  python -m nslkdd.audit notebooks/*.ipynb
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def _cramers_v(x, y):
    table = pd.crosstab(x, y).to_numpy()
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * max(1, min(r - 1, k - 1)))))


def _correlation_ratio(values, y):
    """Eta: fração da variância do atributo numérico explicada pela classe."""
    values = pd.Series(values, dtype=float)
    groups = values.groupby(np.asarray(y))
    ss_between = (groups.count() * (groups.mean() - values.mean()) ** 2).sum()
    ss_total = ((values - values.mean()) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def association_with_target(df, y, alert=0.9, random_state=42):
    """Tabela atributo x (eta/Cramér V, informação mútua) ordenada; flag >= alert."""
    rows = []
    for col in df.columns:
        x = df[col]
        if not pd.api.types.is_numeric_dtype(x):
            strength, kind = _cramers_v(x, y), "cramers_v"
            mi_input = x.astype("category").cat.codes.to_frame()
            discrete = True
        else:
            strength, kind = _correlation_ratio(x, y), "eta"
            mi_input, discrete = x.to_frame(), False
        mi = mutual_info_classif(mi_input, y, discrete_features=discrete,
                                 random_state=random_state)[0]
        rows.append({"atributo": col, "medida": kind, "associacao": round(strength, 4),
                     "info_mutua": round(float(mi), 4), "n_valores": x.nunique()})
    out = pd.DataFrame(rows).sort_values("associacao", ascending=False, ignore_index=True)
    out["alerta"] = out["associacao"] >= alert
    return out


PATTERNS = [
    ("ERRO", r"\.(fit|fit_transform|fit_resample|partial_fit)\([^)]*\b\w*test\b",
     "Ajuste (fit) usando dados de teste."),
    ("ERRO", r"concat\([^)]*test", "Concatenação treino+teste antes do pré-processamento."),
    ("ALERTA", r"train_test_split\(", "Re-split do teste é proibido: confirme que é um "
     "holdout estratificado só dentro do KDDTrain+."),
    ("ERRO", r"train_test_split\([^)]*\b\w*test\b", "train_test_split envolvendo o KDDTest+."),
    ("ALERTA", r"['\"]difficulty['\"]", "Atributo 'difficulty' citado: deve ser descartado."),
    ("ALERTA", r"\bfit_resample\(", "Reamostragem fora do Pipeline: se feita antes da CV, "
     "os folds de validação recebem amostras sintéticas (vazamento)."),
    ("ALERTA", r"(Scaler|Encoder|Imputer|PCA|SelectKBest)\([^)]*\)\.fit(_transform)?\(",
     "Transformador ajustado fora do Pipeline: confirme que não é antes da CV."),
    ("ALERTA", r"(?<!def )load_test(_frame)?\(", "Uso do KDDTest+: só permitido no notebook 04."),
]


def _sources(path: Path):
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"], 1):
            if cell["cell_type"] == "code":
                yield f"célula {i}", "".join(cell["source"])
    else:
        yield "arquivo", path.read_text(encoding="utf-8")


def scan_notebook(path):
    path = Path(path)
    final_stage = path.name.startswith("04")
    findings = []
    for where, src in _sources(path):
        for n, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for level, pattern, msg in PATTERNS:
                if final_stage and "load_test" in pattern:
                    continue
                if re.search(pattern, line):
                    findings.append({"arquivo": path.name, "local": f"{where}, linha {n}",
                                     "nivel": level, "problema": msg, "codigo": line.strip()[:120]})
    return findings


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    all_findings = [f for p in paths for f in scan_notebook(p)]
    for f in all_findings:
        print(f"[{f['nivel']}] {f['arquivo']} ({f['local']}): {f['problema']}\n    {f['codigo']}")
    errors = sum(f["nivel"] == "ERRO" for f in all_findings)
    print(f"\n{len(all_findings)} achado(s), {errors} erro(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
