---
name: etapa3-modelagem
description: Etapa 3 (entrega 21/11) — treinar e comparar ≥3 algoritmos da disciplina (Árvore de Decisão, Random Forest, XGBoost/boosting, MLP, KNN, Naive Bayes) com as melhores combinações de pré-processamento, ajustar hiperparâmetros por CV no treino, escolher o modelo final e congelar results/final_config.json. Use para tuning, comparação de modelos, ensembles, ajuste de pesos de decisão para F1-macro e estratégia para subir no ranking.
---

# Etapa 3 — Construção e avaliação dos classificadores

Entregável: `notebooks/03_modelagem.ipynb` + `results/final_config.json`. Requisito 6: ≥3 algoritmos **vistos em disciplina** (material: Árvores de Decisão, Bagging, Boosting/AdaBoost, Stacking, XGBoost, Random Forest; o enunciado cita RN).

## Algoritmos recomendados e espaços de busca
| Modelo | Por que incluir | Hiperparâmetros principais |
|---|---|---|
| `DecisionTreeClassifier` | baseline interpretável (material Partes 1–5) | `max_depth` [8,12,16,24,None], `min_samples_leaf` [1,2,5,10], `criterion` [gini, entropy], `class_weight` [None, balanced] |
| `RandomForestClassifier` | bagging + atributos aleatórios (slide: m≈√p) | `n_estimators` [200,400], `max_features` [sqrt, log2, 0.3], `min_samples_leaf` [1,2,4], `class_weight` [None, balanced, balanced_subsample] |
| `XGBClassifier` (xgboost) | boosting citado no material | envolver em `nslkdd.models.EncodedLabelClassifier(XGBClassifier(tree_method="hist", n_jobs=1))` (XGB exige rótulos 0..k-1); busca com prefixo `model__estimator__`: `n_estimators` [300,600], `max_depth` [4,6,8], `learning_rate` [0.05,0.1], `subsample`/`colsample_bytree` [0.8,1.0]. Desbalanceamento por reamostragem (sample_weight não atravessa samplers). Smoke: 200 árvores + SMOTE → CV 0,943, ~38 s |
| `MLPClassifier` (RN) | requer escala → valida a escolha de normalização | `hidden_layer_sizes` [(64,), (128,64)], `alpha` [1e-4,1e-3], `early_stopping=True`; não aceita class_weight → usar balanceamento por reamostragem |
| `KNeighborsClassifier` (opcional) | sensível a escala/dimensionalidade | `n_neighbors` [3,5,9], `weights` [uniform, distance] — lento na predição, usar após redução |

Tuning: `RandomizedSearchCV(build_pipeline(...), param_distributions={"model__max_depth": [...], ...}, scoring="f1_macro", cv=make_cv()[0], n_iter=20, random_state=42)` ou `HalvingRandomSearchCV` para RF/XGB. Prefixo `model__` para o estimador; balanceador é `balance__`. Registrar o resultado do melhor de cada algoritmo com `ExperimentLog.run(stage="etapa3", ...)` para entrar na tabela comparativa.

## Tabela comparativa (20% da nota)
Linhas = (estratégia de pré-processamento × algoritmo); colunas = F1-macro (média ± dp), acurácia balanceada, acurácia, F1 por classe, tempo. Cruzar pelo menos as 2–3 melhores estratégias da Etapa 2 com cada algoritmo (≥ 9 linhas). `log.leaderboard()` já gera; exportar com `.to_markdown()`/`.to_latex()` para o relatório. Discussão obrigatória: qual pré-processamento ajuda qual algoritmo e por quê (ex.: escala importa para MLP/KNN, não para árvores).

## Como subir no ranking sem vazar (o que diferencia equipes)
O teste tem 17 ataques inéditos e muito mais R2L/U2R (skill `nslkdd-base`). A CV estratificada satura em ≈0,94 e quase não discrimina; o que importa no teste é **generalizar para R2L/U2R**. Táticas legítimas, todas decididas só com o treino:
1. **CV por grupo de ataque como segundo critério**: `cv, groups = make_cv("attack_group", n_splits=3, attack=attack)` e `log.run(..., cv_kind="attack_group", n_splits=3, attack=attack)`. Mede se o modelo reconhece *tipos de ataque não vistos* (RF baseline: 0,33). É ruidosa; use para desempatar candidatos próximos na CV estratificada, não como métrica única. Explicar no relatório que simula a novidade do KDDTest+ — é um argumento metodológico forte.
2. **Pesos de decisão para F1-macro** (`nslkdd.decision.WeightedDecisionClassifier`): aprende multiplicadores por classe sobre probabilidades out-of-fold do treino e desloca a fronteira para R2L/U2R. Requer `predict_proba`. Comparar com/sem na tabela.
3. **Regularizar**: `min_samples_leaf` maior, `max_depth` limitado, menos atributos de conteúdo ruidosos — modelos que decoram `warezclient` falham em `guess_passwd`/`warezmaster`.
4. **Balanceamento parcial forte para R2L/U2R** (`min_count` 2000–5000) costuma ajudar mais no teste do que na CV.
5. **Ensemble** (Voting soft RF + XGB + MLP, ou Stacking — ambos no material): diversidade de erros (slide "condições para bom desempenho").
6. Estabilidade: repetir o finalista com 3 sementes; relatar média ± dp.
Proibido: escolher entre candidatos olhando o KDDTest+, ajustar pesos/limiar no teste, usar a distribuição de classes do teste como prior.

## Escolha e congelamento do modelo final
Critério declarado *antes*: maior F1-macro na CV estratificada; empate (diferença < 1 dp) → melhor CV por grupo de ataque → mais simples/explicável. Ao final do notebook:
```python
import json; from nslkdd import config
final = {"model": "RF", "model_params": {...}, "pipeline_params": {...},
         "decision_weights": True, "cv_f1_macro": 0.0, "cv_attack_group_f1": 0.0,
         "criterio": "maior F1-macro CV estratificada 5-fold; desempate por CV por grupo de ataque",
         "data_congelamento": "AAAA-MM-DD"}
(config.RESULTS / "final_config.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
```
Depois disso, nada da Etapa 2/3 é alterado com base no teste.

## Orçamento de tempo (máquina local, 126k linhas)
DT ≈ 8 s/CV, RF-100 ≈ 20–25 s/CV, SMOTE adiciona pouco; MLP e KNN são os mais caros. Para buscas grandes: rodar em background (`run_in_background`) e acompanhar `results/experiments.csv`.
