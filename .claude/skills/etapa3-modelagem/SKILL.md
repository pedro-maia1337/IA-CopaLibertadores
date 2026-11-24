---
name: etapa3-modelagem
description: Etapa 3 (entrega 21/11) — treinar e comparar ≥3 algoritmos da disciplina (Árvore de Decisão, Random Forest, boosting/XGBoost, MLP, KNN, Naive Bayes) com as melhores combinações de pré-processamento, ajustar hiperparâmetros por CV agrupada no treino, escolher o modelo final e congelar results/final_config.json. Use para tuning, comparação de modelos, ensembles, ajuste do limiar de decisão para F1-macro e estratégia para subir no ranking.
---

# Etapa 3 — Construção e avaliação dos classificadores

Entregável: `notebooks/03_modelagem.ipynb` + `results/final_config.json`. Requisito 6: ≥3 algoritmos **vistos em disciplina** (material: Árvores de Decisão, Bagging, Boosting/AdaBoost, Stacking, XGBoost, Random Forest; o enunciado cita RN).

`tools/rodar_experimentos.py etapa3` já executou tudo abaixo, usando `results/melhores_configuracoes.json` da Etapa 2. Os resultados estão em `results/experiments.csv`, e o cache permite que o notebook repita os mesmos `log.run` sem recalcular.

## Comparação de algoritmos (CV agrupada 5-fold, padrão do catálogo `make_model`, F1-macro em todas as linhas)
| Modelo | Pré-processamento | F1-macro |
|---|---|---|
| `HGB_native` | drop_inconsistent, NaN nativo, categorias nativas | 0,8821 ± 0,0009 |
| `RF` | median, onehot | 0,8808 ± 0,0018 |
| `XGB` | igual ao HGB | 0,8799 ± 0,0008 |
| `ET` | igual ao RF | 0,8774 ± 0,0028 |
| `MLP` | median, onehot, standard + log1p | 0,8711 ± 0,0044 |
| `KNN` | igual ao MLP + kbest_f 30 | 0,8624 ± 0,0058 |
| `DT` | igual ao RF | 0,8403 ± 0,0024 |
| `NB` | igual ao MLP | 0,7662 ± 0,0292 |

## Finalistas (CV agrupada 5-fold × 3 repetições) — `results/tabela_etapa3_finalistas.md`
Maior média: voto HGB + XGB + RF, 0,8826 ± 0,0029. Onze candidatos empatam com ele (< 1 dp). **Escolhido pela regra de simplicidade: `HGB busca 2`, 0,8819 ± 0,0030** (sem conflitos, contraste: 0,9817; precisão anomaly 0,759, recall 0,798; limiar médio 0,04). Fora do empate: XGB 0,8790, CatBoost 0,8770, MLP 0,8716. Congelado em `results/final_config.json` (hiperparâmetros `min_samples_leaf=20, max_leaf_nodes=63, max_iter=500, max_features=0.7, learning_rate=0.08, l2_regularization=0.0`; `redundancy="drop_inconsistent"`).

Teto teórico da métrica nesta base: 0,905 (acerto perfeito fora dos conflitos e maioria em cada conflito, empate → `anomaly`). No OOF do escolhido, 1.706 dos 1.851 erros estão nas 3.730 linhas em conflito.

## O que os números dizem
- **Tuning não rende.** Os padrões já estavam no platô: no HGB o melhor sorteio (0,8817) fica abaixo do padrão (0,8821); RF e XGB empatam. A busca foi curta (6–12 sorteios). Diga isso; não apresente como ganho.
- **O limiar é baixo (0,03–0,11 no HGB) por causa dos conflitos:** nos 1.376 grupos empatados, prever `anomaly` rende mais F1-macro. A curva é um platô (0,01–0,39 a menos de 0,002 do máximo).
- **Ensemble não ajudou:** os erros dos modelos fortes são os mesmos.
- **DT 0,84 e NB 0,77 sofrem** com o ruído de rótulo e com atributos correlacionados: boa discussão para a tabela comparativa.
- **Custo do rigor:** as colunas de vazamento, sozinhas, dariam 0,889 (acima de qualquer modelo de tráfego). Abrir mão delas é a decisão correta (ver `anti-vazamento`).
- O F1-macro antigo de 0,983 era calculado só nas linhas únicas e sem conflito (máscara); não é a métrica oficial e não deve ser reportado como desempenho.

## Como subir no ranking sem vazar
Quase tudo que rendia já foi colhido. Em ordem de potencial:
1. **Engenharia de atributos** de janela/razão, inserida como passo sem estado no pipeline. Testado: não ganhou na métrica oficial (CV 3×5).
2. **Média de sementes** do HGB: estabilidade, não ganho esperado.
3. **Estudo dos erros OOF:** fora dos conflitos, 81 falsos negativos, vários `service=private` e `flag=SF`, provavelmente sobrepostos ao tráfego normal.
Proibido: escolher entre candidatos olhando o teste, ajustar limiar pela proporção prevista no teste, pseudo-rótulos, usar `data/raw/`, usar as colunas de vazamento.

## Tabela comparativa (20% da nota)
Linhas = (estratégia de pré-processamento × algoritmo); colunas = F1-macro (média ± dp), F1 anomaly, precisão, recall, acurácia balanceada, tempo. Use `log.leaderboard("etapa3")` e `log.leaderboard("etapa3_final")`; exporte com `.to_markdown()`/`.to_latex()` para o relatório. Discussão obrigatória: qual pré-processamento ajuda qual algoritmo e por quê (ex.: limpeza de rótulo ajuda boosting e não RF; escala importa para MLP/KNN, não para árvores).

## Busca de hiperparâmetros
Use `ExperimentLog.run` em laço com `ParameterSampler` (ver `tuned` em `tools/rodar_experimentos.py`). Não use `GridSearchCV` direto: exigiria passar `groups` e não aplica o limiar cruzado. Ensembles: `nslkdd.models.SoftVote` com `prebuilt=True`; o `VotingClassifier` do sklearn não aceita o `Pipeline` do imblearn. `StackingClassifier` usa CV interna sem grupos, então evite.

## Escolha e congelamento do modelo final
Critério declarado *antes*: maior F1-macro na CV agrupada (`f1_macro_mean`); empate (diferença < 1 dp) → mais simples/explicável. Ao final do notebook, a equipe congela:
```python
import json; from nslkdd import config
final = {"model": "HGB_native",
         "model_params": {"min_samples_leaf": 20, "max_leaf_nodes": 63, "max_iter": 500,
                          "max_features": 0.7, "learning_rate": 0.08, "l2_regularization": 0.0},
         "pipeline_params": {"redundancy": "drop_inconsistent", "domain": "nan", "imputation": "none",
                             "scaling": "none", "log_transform": False,
                             "service_encoding": "ordinal", "categorical_encoding": "ordinal"},
         "threshold": "ThresholdClassifier (OOF no treino, todas as linhas)", "cv_f1_macro": 0.8819,
         "criterio": "maior F1-macro em CV agrupada 5x3; empate < 1 dp pelo mais simples",
         "data_congelamento": "AAAA-MM-DD"}
(config.RESULTS / "final_config.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
```
Depois disso, nada da Etapa 2/3 é alterado.

## Orçamento de tempo (16 núcleos, 70k linhas, 5 folds em paralelo)
HGB ≈ 16 s/CV, XGB ≈ 18 s, MLP ≈ 30 s, RF ≈ 45–60 s, e a busca completa `etapa2` + `etapa3` leva menos de 1 h. Use `threads=3` no modelo e `n_jobs=5` em `ExperimentLog.run`.
