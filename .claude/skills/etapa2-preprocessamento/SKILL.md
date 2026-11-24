---
name: etapa2-preprocessamento
description: Desenho e execução dos experimentos da Etapa 2 (entrega 07/11) — comparar estratégias de redundância/inconsistência, valores impossíveis, imputação de ausentes, ≥2 técnicas de balanceamento, codificação simbólico-numérica de service, normalização/transformação numérica e redução de dimensionalidade ou seleção de instâncias, tudo via CV agrupada no treino. Use ao planejar, rodar ou justificar qualquer escolha de pré-processamento.
---

# Etapa 2 — Experimentos de pré-processamento

Entregável: `notebooks/02_preprocessamento.ipynb`. Requisitos 1–5 da Seção 4 do enunciado, mais dados ausentes: **todas as técnicas devem ser testadas e comparadas explicitamente**, não basta apresentar a escolhida.

## Ferramentas
```python
from nslkdd import load_train, ExperimentLog
from nslkdd.models import make_model
X, y, groups = load_train()
log = ExperimentLog()                       # results/experiments.csv, com cache
log.run("etapa2_redundancia", "RF", make_model("RF", threads=3), X, y, groups, params={...}, n_jobs=5)
log.leaderboard("etapa2_redundancia")      # tabela comparativa pronta
```
`tools/rodar_experimentos.py etapa2` já executou a busca abaixo. Como o log tem cache, o notebook 02 pode chamar os mesmos `log.run` e receber os resultados sem recalcular.

Parâmetros de `build_pipeline` (cada um é um eixo de comparação):
| Eixo | Valores | Requisito |
|---|---|---|
| `redundancy` | none, drop_duplicates, drop_inconsistent, majority_label, relabel_model | 1 |
| `domain` | nan, clip, none | 1 (inconsistências) |
| `imputation` | none (NaN nativo), mean, median, median_indicator, iterative | ausentes |
| `balancing` | none, class_weight, ros, rus, smote, borderline_smote, adasyn, smote_tomek, smote_enn, hybrid, tomek, enn | 2 (+ seleção de instâncias) |
| `min_count` / `max_count` | alvo do over/undersampling (ex.: 10000 / 20000) | 2 |
| `service_encoding`, `categorical_encoding` | onehot, onehot_rare, frequency, target, ordinal | 3 |
| `scaling` | standard, minmax, robust, quantile, none | 4 |
| `log_transform` | True/False (log1p com sinal nos `config.HEAVY_TAILED`) | 4 |
| `reduction` / `reduction_k` | none, pca, kbest_f, kbest_mi, tree_select | 5 |

`imputation="none"` só serve para árvores/boosting, e `build_pipeline` recusa combiná-la com reamostragem ou redução. `HGB_native` exige `service_encoding` e `categorical_encoding` iguais a `ordinal`.

## Protocolo (eficiente e defensável)
Busca **um eixo por vez** (coordinate search), com o resto fixo na melhor escolha até então, por família: HGB, RF e MLP. O MLP é necessário porque árvores são invariantes a escala; sem ele a comparação de normalização é vazia (dizer isso no texto).

**Regra de decisão pré-declarada:** a primeira opção do eixo é a mais simples, e uma alternativa só a substitui se a superar em mais de 1 desvio-padrão. Diferença menor não é diferença.

Métrica: `f1_macro_mean` (F1-macro em **todas as linhas** do fold, como a métrica oficial; limiar cruzado). Os rótulos contraditórios a limitam a 0,905 mesmo para um classificador perfeito; os modelos ficam perto de 0,88. `f1_macro_clean_mean` (linhas únicas e sem conflito, mesmo limiar) fica como contraste e não decide nada. Até 24/11/2026 a métrica principal era a de linhas limpas (~0,98), que não é a oficial; o registro antigo está em `results/arquivo/`.

## Resultados medidos (CV agrupada 5-fold, F1-macro em todas as linhas ± dp)
| Eixo | HGB nativo | RF | MLP |
|---|---|---|---|
| **Redundância** | none 0,8777 ± 0,0025; drop_duplicates 0,8779; **drop_inconsistent 0,8821 ± 0,0009**; majority_label 0,8815; relabel_model 0,8814 | **none 0,8808 ± 0,0018**; drop_duplicates 0,8816; drop_inconsistent 0,8799; majority 0,8812; relabel 0,8802 | **none 0,8711 ± 0,0044**; drop_duplicates 0,8666; drop_inconsistent 0,8696 |
| **Domínio** | **nan 0,8821**; none 0,8817; clip 0,8814 | não testado | **nan 0,8711**; none 0,8681; clip 0,8699 |
| **Imputação** | **none 0,8821**; median 0,8817; median_indicator 0,8814 | **median 0,8808**; none 0,8809; median_indicator 0,8807; mean 0,8811 | **median 0,8711**; mean 0,8692; median_indicator 0,8594; iterative 0,8685 |
| **Codificação de `service`** | nativa (fixa) | **onehot 0,8808**; onehot_rare 0,8812; frequency 0,8803; target 0,8800; ordinal 0,8801 | **onehot 0,8711**; onehot_rare 0,8730; frequency 0,8675; target 0,8651 |
| **Balanceamento** | **none 0,8821**; class_weight 0,8807 | **none 0,8808**; class_weight 0,8763; ROS 0,8799; RUS 0,8746; SMOTE 0,8807; borderline 0,8810; SMOTE+Tomek 0,8816; híbrido 0,8794 | **none 0,8711**; ROS 0,8693; RUS 0,8664; SMOTE 0,8680; borderline 0,8548; SMOTE+Tomek 0,8702; híbrido 0,8704 |
| **Escala** | não se aplica | não se aplica | **standard+log 0,8711**; standard 0,8654; minmax+log 0,8669; robust+log 0,8623; quantile 0,8646 |
| **Redução** | não testada | **none 0,8808**; PCA 0,7201 ± 0,0537; kbest_f 0,8639; kbest_mi 0,8785; tree_select 0,8819 ± 0,0016 | **none 0,8711**; PCA 0,8655; kbest_f 0,8600; kbest_mi 0,8641; tree_select 0,8724 |

Em negrito, a opção escolhida pela regra. Valores no `results/experiments.csv`. Robustez (CV 3×5): HGB partida 0,8777 ± 0,0032 → escolhida 0,8815 ± 0,0027; RF none 0,8809 ± 0,0030 × tree_select 0,8812 (empate).

## O que os números dizem
- **O ganho real vem da limpeza de rótulo**, e só no boosting: HGB sobe 0,8777 → 0,8821 (drop_inconsistent, majority_label e relabel_model empatam entre si). RF é robusto ao ruído por natureza (bagging), e no MLP não houve diferença clara.
- **`relabel_model` não compensou**: recuperar os ~970 ataques dos conflitos não superou a simples remoção. Relate como negativo; é um bom resultado para o relatório.
- **Balanceamento não ajuda aqui.** Com limiar ajustado por OOF, o desbalanceamento 16:1 já está tratado na decisão. Nenhuma técnica passou da regra, e RUS e `class_weight` pioram. Cumpra o requisito 2 comparando (ROS, RUS, SMOTE) e justifique a escolha de `none`.
- **`log1p` importa para o MLP** (0,8654 → 0,8711), e a escala importa: robust perde 0,009.
- **PCA no RF (0,72) é artefato:** com `scaling="none"`, os componentes ficam dominados pela variância de `dst_bytes`. Sempre explique ao apresentar, ou compare PCA só com dados padronizados (RF 0,8686; MLP 0,8655).
- **Valores impossíveis:** `nan` é ao menos tão bom quanto as alternativas em HGB e MLP.

## Justificativas técnicas (use, adapte, cite o material)
- **Redundância**: cópias inflam o peso de padrões repetidos; inconsistentes são ruído de rótulo (o slide define "duas instâncias iguais com classificações diferentes"). `majority_label` usa a maioria quando há três cópias e descarta os empates. `relabel_model` usa um modelo treinado no fold para decidir os conflitos. As três limpezas empatam no HGB; fica `drop_inconsistent`, a primeira que passa da regra. Os conflitos saem só do treino de cada fold: a validação continua com todas as linhas.
- **Valores impossíveis**: não há como saber o fator aplicado (600 pode ser 6×100 ou 60×10), então tratar como ausente evita inventar um valor.
- **Ausentes**: slide Parte 2. Descartar linhas é inviável (94% têm algum ausente) e o teste também os tem. `dst_host_serror_rate` é MNAR, o que favorece NaN nativo ou indicador. Note que o indicador não melhorou o resultado medido; relate isso.
- **Balanceamento**: slide Parte 1 — oversampling arrisca overfitting/instâncias irreais; undersampling arrisca underfitting/perda de informação. Batista et al. (2004, no material) recomendam SMOTE+Tomek para poucos positivos: testado. SMOTE roda depois do one-hot/escala e gera dummies fracionárias (limitação a mencionar).
- **Codificação de `service`** (requisito 3 pede discutir alta cardinalidade): one-hot preserva ausência de ordem (slide: distância de Hamming constante) mas cria 67 colunas esparsas; o slide alerta que 1-de-c não é boa opção para muitas categorias ("193 países") e sugere pseudoatributos → `frequency` e `target`; `onehot_rare` agrupa raras; `ordinal` impõe ordem falsa, aceitável só para árvores; `HGB_native` trata a categoria sem ordem.
- **Transformação numérica**: `log1p` comprime caudas longas sem eliminar instâncias. Standard vs MinMax vs Robust importa para MLP/KNN/SMOTE (distâncias), não para árvores.
- **Redução**: nenhuma passou da regra; `tree_select` empata com não reduzir (CV 3×5) e manteria os atributos originais; PCA perde informação discriminante e interpretabilidade.
- **Outliers**: não concentram anomalias nesta base (4,7% vs 6,0%), mas removê-los perderia 11,8% das anomalias. Valores impossíveis são outro caso, já tratado.

## Armadilhas
- Não rodar `fit_resample` fora do pipeline (ver `anti-vazamento`).
- Sobrecarga de núcleos: use `ExperimentLog.run(..., n_jobs=5)` com `threads=3` no modelo. São 16 núcleos.
- ADASYN pode falhar em classes sem vizinhos difíceis — capture e registre.
- Experimentos já registrados são pulados; use `force=True` se mudou o código do pré-processamento.
- Não confie em conclusões herdadas do NSL-KDD original: aqui o F1 é binário e o desbalanceamento já é resolvido pelo limiar.
