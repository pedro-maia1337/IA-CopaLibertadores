---
name: etapa1-eda
description: Roteiro da Etapa 1 (entrega 24/10) — notebook de análise exploratória e diagnóstico do NSL-KDD: classes, desbalanceamento, atributos categóricos, inconsistências, redundâncias e possíveis vazamentos. Use ao criar, ampliar ou revisar notebooks/01_eda.ipynb ou ao escrever o diagnóstico dos dados.
---

# Etapa 1 — Análise exploratória e diagnóstico

Entregável: `notebooks/01_eda.ipynb` (gerado de `notebooks/src/01_eda.py`, ver `notebook-padrao`). Fatos de referência para conferir resultados: skill `nslkdd-base`.

**Decisão de escopo sobre o teste:** a EDA é feita no **KDDTrain+**. Uma única seção final, descritiva, compara a *distribuição de rótulos* do teste com a do treino (contagens por classe e lista de ataques inéditos), sem olhar atributos do teste e sem tomar decisão com isso. Deixe isso explícito em markdown. Essa é a única exceção ao bloqueio; use `load_test_frame(final_evaluation=True)[["attack", "category"]]` e documente o motivo.

## Estrutura do notebook (seções numeradas)
0. **Cabeçalho**: título, equipe, objetivo, versão das bibliotecas, semente, declaração de que o teste não é usado para decisões.
1. **Carga e visão geral**: `load_train_frame()`; shape, `dtypes`, `head`, memória; tabela tipo de cada atributo (categórico/binário/contínuo/taxa).
2. **Valores ausentes**: `isna().sum()` (zero) + checagem de valores impossíveis (taxas fora de [0,1], negativos, `su_attempted == 2`, binários com >2 valores).
3. **Classe-alvo e desbalanceamento**: contagem e % por `category` e por `attack`; gráfico em escala log; razão de desbalanceamento (normal/U2R ≈ 1.295:1); por que acurácia engana (classificador "tudo normal" tem 53% de acurácia e F1-macro ≈ 0,14).
4. **Atributos categóricos**: cardinalidade; distribuição de `protocol_type` e `flag` por classe (crosstab normalizado); top-20 `service` + cauda; serviços raros (< 0,1%); discussão da alta cardinalidade de `service` (one-hot = 70 colunas esparsas vs frequência/target encoding) → alimenta a Etapa 2.
5. **Atributos numéricos**: `describe()`; assimetria (`skew`) e % de zeros; histogramas log de `src_bytes`, `dst_bytes`, `duration`, `count`; boxplots por classe dos ~8 mais discriminantes; constantes (`num_outbound_cmds`) e quase constantes (`is_host_login`, `land`, `urgent`).
6. **Outliers**: IQR/z-score em escala log; mostrar que "outliers" de `src_bytes` concentram ataques → remover outliers aqui destruiria sinal das classes raras (argumento para *não* eliminar outliers por regra cega ou para fazê-lo só por classe).
7. **Redundância de atributos**: heatmap de correlação (Pearson e Spearman) dos numéricos; listar pares |r| > 0,95 (há 10, ver `nslkdd-base/references/atributos.md`).
8. **Redundância e inconsistência de instâncias**: duplicatas exatas (com e sem `difficulty`), grupos inconsistentes (mesmos 41 atributos, rótulos diferentes) com exemplos; estratégias candidatas para a Etapa 2 (manter / remover duplicatas / remover inconsistentes / rótulo majoritário).
9. **Investigação de vazamento (Seção 7 do enunciado)**: `association_with_target` (eta para numéricos, V de Cramér para categóricos, informação mútua) incluindo `difficulty`; tabela ordenada + barra; conclusão: remover `difficulty` (metadado de rotulagem), nenhum dos 41 atributos é proxy do rótulo.
10. **Comparação descritiva treino × teste (só rótulos)**: tabela de classes; 17 ataques inéditos; R2L 995→2.754 e U2R 52→200; implicação: CV no treino será otimista.
11. **Diagnóstico e plano para a Etapa 2**: tabela "problema encontrado → evidência → técnica que será testada → por quê" (ver modelo abaixo).

## Modelo da tabela de diagnóstico (célula final)
| Problema | Evidência | Técnica(s) a comparar na Etapa 2 |
|---|---|---|
| Desbalanceamento extremo | U2R 0,04%, R2L 0,8% | SMOTE, ROS, RUS, híbrido, class_weight |
| `service` com 70 categorias | 70 valores, cauda longa | one-hot vs one-hot com raras agrupadas vs frequência vs target |
| Cauda longa / escalas distintas | skew de `src_bytes` ≫ 10 | log1p + Standard/MinMax/Robust |
| Atributo constante | `num_outbound_cmds` | remover |
| Atributos redundantes | 10 pares r > 0,95 | seleção (SelectKBest/árvores) ou PCA |
| Instâncias inconsistentes | 15 grupos | 4 estratégias de redundância |
| Vazamento | `difficulty` | descartar |
| Mudança de distribuição | 17 ataques inéditos | CV por grupo de ataque como diagnóstico |

## Qualidade
- Cada gráfico com título, eixos e 1–2 frases de interpretação logo abaixo (a professora avalia o *diagnóstico*, não a quantidade de gráficos).
- Salvar figuras relevantes em `results/figures/eda_*.png` (servem ao relatório).
- Rodar `python -m nslkdd.audit notebooks/01_eda.ipynb` — o único `load_test` permitido é o da seção 10; justifique-o em markdown.
- Use os termos do material da disciplina (balanceamento, dados redundantes/inconsistentes, conversão simbólico-numérica); para citar, `Grep` em `docs/_txt/material/`.
