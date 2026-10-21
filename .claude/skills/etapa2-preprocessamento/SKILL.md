---
name: etapa2-preprocessamento
description: Desenho e execução dos experimentos da Etapa 2 (entrega 07/11) — comparar estratégias de redundância/inconsistência, ≥2 técnicas de balanceamento, codificação simbólico-numérica de service, normalização/transformação numérica e redução de dimensionalidade ou seleção de instâncias, tudo via CV no treino. Use ao planejar, rodar ou justificar qualquer escolha de pré-processamento.
---

# Etapa 2 — Experimentos de pré-processamento

Entregável: `notebooks/02_preprocessamento.ipynb`. Requisitos 1–5 da Seção 4 do enunciado: **todas as técnicas devem ser testadas e comparadas explicitamente**, não basta apresentar a escolhida.

## Ferramentas
```python
from nslkdd import load_train, build_pipeline, ExperimentLog
X, y, attack = load_train()
log = ExperimentLog()                       # results/experiments.csv, com cache
log.run("etapa2", "RF", rf, X, y, params={...})
log.leaderboard("etapa2")                  # tabela comparativa pronta
```
Parâmetros de `build_pipeline` (cada um é um eixo de comparação):
| Eixo | Valores | Requisito |
|---|---|---|
| `redundancy` | none, drop_duplicates, drop_inconsistent, majority_label | 1 |
| `balancing` | none, class_weight, ros, rus, smote, borderline_smote, adasyn, smote_tomek, smote_enn, hybrid, tomek, enn | 2 (+ seleção de instâncias) |
| `min_count` / `max_count` | alvo do over/undersampling (ex.: 2000 / 20000) | 2 |
| `service_encoding` | onehot, onehot_rare, frequency, target, ordinal | 3 |
| `scaling` | standard, minmax, robust, quantile, none | 4 |
| `log_transform` | True/False (log1p nos `config.HEAVY_TAILED`) | 4 |
| `reduction` / `reduction_k` | none, pca, kbest_f, kbest_mi, tree_select | 5 |

## Protocolo (eficiente e defensável)
Busca **um eixo por vez** (coordinate search), com o resto fixo na melhor escolha até então. Evita grade combinatória de horas e gera tabelas legíveis.
1. **Modelos-sonda**: `DecisionTreeClassifier(random_state=42)` (rápido, ~8 s/CV) e `RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)` (~20 s/CV). Opcional: um modelo sensível a escala (KNN ou MLP) nos eixos 4 e 5, porque árvores são invariantes a escala — sem ele a comparação de normalização é vazia (dizer isso no texto!).
2. Ordem: baseline sem nada → redundância → codificação → transformação numérica → balanceamento → redução.
3. Cada eixo: tabela (config × F1-macro ± dp × F1 por classe), gráfico de barras com barras de erro, parágrafo de discussão.
4. Para decidir empates, olhar F1 de **R2L e U2R** e o desvio-padrão; diferença < 1 dp não é diferença — escolher a opção mais simples/mais justificável.
5. Ao final, combinação vencedora + 2–3 combinações alternativas que seguem para a Etapa 3.

## Resultados de referência (smoke test no KDDTrain+, CV estratificada 5-fold)
| Config | F1-macro | F1 U2R |
|---|---|---|
| DT, sem pré-processamento extra | 0,908 | 0,58 |
| RF-100 baseline | 0,913 | 0,60 |
| RF + SMOTE(min 2000) | 0,940 | 0,73 |
| RF + drop_inconsistent + frequency + hybrid(2000/20000) | 0,944 | 0,75 |
Ganhos vêm quase todos de U2R/R2L. Normal/DoS/Probe já saturam ≈ 0,997+.

## Justificativas técnicas (use, adapte, cite o material)
- **Redundância**: duplicatas inflam o peso de padrões repetidos e, entre folds, inflam a CV; inconsistentes são ruído de rótulo (o slide define "duas instâncias iguais com classificações diferentes"). `majority_label` preserva informação; `drop_inconsistent` é mais conservador. Volume pequeno (16/15) → efeito esperado pequeno; relatar mesmo assim.
- **Balanceamento**: slide Parte 1 — oversampling arrisca overfitting/instâncias irreais; undersampling arrisca underfitting/perda de informação. `rus` "auto" reduz tudo a 52 por classe → descarta 99,9% dos dados (mostrar que piora). Por isso usar estratégias **parciais** (`min_count`, `max_count`) e o `hybrid`. Batista et al. (2004, no material) recomendam SMOTE+Tomek/SMOTE+ENN para poucos positivos — testar `smote_tomek`. SMOTE roda depois do one-hot/escala: gera dummies fracionárias (limitação a mencionar; alternativa SMOTENC).
- **Codificação de `service`** (requisito 3 pede discutir alta cardinalidade): one-hot preserva ausência de ordem (slide: distância de Hamming constante) mas cria 70 colunas esparsas; slide alerta que 1-de-c não é boa opção para muitas categorias ("193 países") e sugere pseudoatributos → `frequency` e `target` são pseudoatributos numéricos; `onehot_rare` agrupa raras; `ordinal` impõe ordem falsa (aceitável só para árvores — explicar). `protocol_type` (3) e `flag` (11) ficam em one-hot sempre.
- **Transformação numérica**: `log1p` comprime caudas longas (bytes, contagens) e reduz influência de extremos sem eliminar outliers (que carregam sinal de ataque, ver EDA). Standard vs MinMax vs Robust importa para KNN/MLP/SMOTE (distâncias), não para árvores.
- **Redução**: PCA (95% da variância) remove a redundância dos grupos serror/rerror mas perde interpretabilidade (ruim para a explicabilidade da Etapa 4); `kbest_mi`/`tree_select` mantêm atributos originais. Alternativa "seleção de instâncias": `tomek`/`enn` como limpeza de fronteira. Escolher **um** para cumprir o requisito 5 e comparar ≥2 variantes dele.
- **Outliers**: se testar eliminação, fazer por classe e só no treino; mostrar que remover outliers globais apaga U2R/R2L.

## Armadilhas
- Não rodar `fit_resample` fora do pipeline (ver `anti-vazamento`).
- `n_jobs=-1` no `cross_validate` + `n_jobs=-1` no RF = oversubscription; use paralelismo em um nível só (`ExperimentLog.run(..., n_jobs=-1)` com RF `n_jobs=1`).
- SMOTE/ENN/Tomek com k-vizinhos em 100k+ linhas de alta dimensão podem levar minutos; ADASYN pode falhar em classes sem vizinhos difíceis — capture e registre.
- Experimentos já registrados são pulados; use `force=True` só se mudou código do pré-processamento.
