---
name: etapa3-modelagem
description: Etapa 3 (entrega 21/11) — treinar e comparar ≥3 algoritmos da disciplina (Árvore de Decisão, Random Forest, boosting/XGBoost, MLP, KNN, Naive Bayes) com as melhores combinações de pré-processamento, ajustar hiperparâmetros por CV agrupada no treino, escolher o modelo final e congelar results/final_config.json. Use para tuning, comparação de modelos, ensembles, ajuste do limiar de decisão para F1-macro e estratégia para subir no ranking.
---

# Etapa 3 — Construção e avaliação dos classificadores

Entregável: `notebooks/03_modelagem.ipynb` + `results/final_config.json`. Requisito 6: ≥3 algoritmos **vistos em disciplina** (material: Árvores de Decisão, Bagging, Boosting/AdaBoost, Stacking, XGBoost, Random Forest; o enunciado cita RN).

`tools/rodar_experimentos.py etapa3` já executou tudo abaixo, usando `results/melhores_configuracoes.json` da Etapa 2. Os resultados estão em `results/experiments.csv`, e o cache permite que o notebook repita os mesmos `log.run` sem recalcular.

## Comparação de algoritmos (CV agrupada 5-fold, padrão do catálogo `make_model`)
| Modelo | Pré-processamento | F1-macro |
|---|---|---|
| `HGB_native` | majority_label, NaN nativo, categorias nativas | 0,9820 ± 0,0015 |
| `RF` | median, onehot, tree_select | 0,9813 ± 0,0020 |
| `XGB` | igual ao HGB | 0,9786 ± 0,0013 |
| `ET` | igual ao RF | 0,9741 ± 0,0045 |
| `MLP` | median, onehot, standard + log1p | 0,9641 ± 0,0038 |
| `KNN` | igual ao MLP + kbest_f 30 | 0,9518 ± 0,0083 |
| `DT` | igual ao RF | 0,8937 ± 0,0045 |
| `NB` | igual ao MLP | 0,7763 ± 0,0419 |

## Finalistas (CV agrupada 5-fold × 3 repetições, com busca de hiperparâmetros)
| Modelo | F1-macro | F1 limiar 0,5 | F1 bruto | F1 anomaly | Precisão | Recall | Limiar | Tempo |
|---|---|---|---|---|---|---|---|---|
| **HGB_native** | **0,9823 ± 0,0023** | 0,9809 | 0,8819 | 0,9658 | 0,9783 | 0,9538 | 0,26 | 47 s |
| Voto HGB+XGB+RF | 0,9821 ± 0,0026 | 0,9813 | 0,8819 | 0,9653 | 0,9753 | 0,9557 | 0,31 | 638 s |
| RF | 0,9802 ± 0,0019 | 0,9787 | 0,8806 | 0,9618 | 0,9755 | 0,9485 | 0,41 | 437 s |
| XGB | 0,9780 ± 0,0027 | 0,9784 | 0,8793 | 0,9575 | 0,9727 | 0,9431 | 0,31 | 52 s |
| MLP | 0,9640 ± 0,0048 | 0,9623 | 0,8705 | 0,9303 | 0,9461 | 0,9156 | 0,42 | 160 s |

**Candidato: `HGB_native`.** Pelo critério abaixo, o voto empata (diferença < 1 dp) e perde em simplicidade e explicabilidade. Hiperparâmetros: `min_samples_leaf=5, max_leaf_nodes=15, max_iter=800, learning_rate=0.06, l2_regularization=0.1`.

## O que os números dizem
- **Tuning quase não rende.** Os padrões já estavam no platô: HGB 0,9820 antes e depois; no RF e no XGB, a melhor configuração sorteada ficou até abaixo do padrão (0,9795 vs 0,9813; 0,9781 vs 0,9786). A busca foi curta (6–12 sorteios). Diga isso; não apresente como ganho.
- **O limiar rende pouco com HGB (0,9809 → 0,9823), mas ele é necessário:** é o que torna o balanceamento dispensável.
- **Ensemble não ajudou:** os erros dos modelos fortes são os mesmos, o que sugere um teto próximo de 0,98 com os atributos de rede.
- **DT 0,89 e NB 0,78 sofrem** com o ruído de rótulo e com a redução: boa discussão para a tabela comparativa.
- **Custo do rigor:** as colunas de vazamento dariam cerca de 0,995. Abrir mão delas é a decisão correta (ver `anti-vazamento`).

## Como subir no ranking sem vazar
Quase tudo que rendia já foi colhido. Em ordem de potencial:
1. **Engenharia de atributos** de janela/razão, inserida como passo sem estado no pipeline. Um teste rápido deu ganho dentro do ruído (0,9795 → 0,9800), então só vale com repetição.
2. **Média de sementes** do HGB: estabilidade, não ganho esperado.
3. **Estudo dos erros OOF:** ~90 falsos negativos, vários `service=private` e `flag=SF`, provavelmente sobrepostos ao tráfego normal.
Proibido: escolher entre candidatos olhando o teste, ajustar limiar pela proporção prevista no teste, pseudo-rótulos, usar `data/raw/`, usar as colunas de vazamento.

## Tabela comparativa (20% da nota)
Linhas = (estratégia de pré-processamento × algoritmo); colunas = F1-macro (média ± dp), F1 anomaly, precisão, recall, acurácia balanceada, tempo. Use `log.leaderboard("etapa3")` e `log.leaderboard("etapa3_final")`; exporte com `.to_markdown()`/`.to_latex()` para o relatório. Discussão obrigatória: qual pré-processamento ajuda qual algoritmo e por quê (ex.: limpeza de rótulo ajuda boosting e não RF; escala importa para MLP/KNN, não para árvores).

## Busca de hiperparâmetros
Use `ExperimentLog.run` em laço com `ParameterSampler` (ver `tuned` em `tools/rodar_experimentos.py`). Não use `GridSearchCV` direto: exigiria passar `groups` e pontuaria também cópias e conflitos. Ensembles: `nslkdd.models.SoftVote` com `prebuilt=True`; o `VotingClassifier` do sklearn não aceita o `Pipeline` do imblearn. `StackingClassifier` usa CV interna sem grupos, então evite.

## Escolha e congelamento do modelo final
Critério declarado *antes*: maior F1-macro na CV agrupada (`f1_macro_mean`); empate (diferença < 1 dp) → mais simples/explicável. Ao final do notebook, a equipe congela:
```python
import json; from nslkdd import config
final = {"model": "HGB_native",
         "model_params": {"min_samples_leaf": 5, "max_leaf_nodes": 15, "max_iter": 800,
                          "learning_rate": 0.06, "l2_regularization": 0.1},
         "pipeline_params": {"redundancy": "majority_label", "domain": "nan", "imputation": "none",
                             "scaling": "none", "log_transform": False,
                             "service_encoding": "ordinal", "categorical_encoding": "ordinal"},
         "threshold": "ThresholdClassifier (OOF no treino)", "cv_f1_macro": 0.9823,
         "criterio": "maior F1-macro em CV agrupada 5x3; empate < 1 dp pelo mais simples",
         "data_congelamento": "AAAA-MM-DD"}
(config.RESULTS / "final_config.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
```
Depois disso, nada da Etapa 2/3 é alterado.

## Orçamento de tempo (16 núcleos, 70k linhas, 5 folds em paralelo)
HGB ≈ 16 s/CV, XGB ≈ 18 s, MLP ≈ 30 s, RF ≈ 45–60 s, e a busca completa `etapa2` + `etapa3` leva menos de 1 h. Use `threads=3` no modelo e `n_jobs=5` em `ExperimentLog.run`.
