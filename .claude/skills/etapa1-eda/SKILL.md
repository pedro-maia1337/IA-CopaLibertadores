---
name: etapa1-eda
description: Roteiro da Etapa 1 — notebook de análise exploratória e diagnóstico da base da competição: classe-alvo, desbalanceamento, atributos categóricos, ausentes, valores impossíveis, redundâncias, inconsistências e vazamentos. Use ao criar, ampliar ou revisar notebooks/01_eda.ipynb ou ao escrever o diagnóstico dos dados.
---

# Etapa 1 — Análise exploratória e diagnóstico

Entregável: `notebooks/01_eda.ipynb` (gerado de `notebooks/src/01_eda.py`, ver `notebook-padrao`). Refeito em 31/10/2026 para a base atualizada. Fatos de referência para conferir resultados: skill `nslkdd-base`.

**Decisão de escopo sobre o teste:** a EDA usa **somente `treino.csv`**. O teste não é lido aqui; como não tem rótulos, não há nem comparação descritiva de classes. A compatibilidade de esquema do teste é verificada em `tests/test_data.py`.

## Estrutura do notebook (seções numeradas)
0. **Cabeçalho**: título, equipe, objetivo, versões, semente, política de dados.
1. **Carga e visão geral**: arquivo bruto e carregador; papel de cada coluna, destacando as 8 não documentadas.
2. **Qualidade dos dados**: grafias, ausentes, mecanismo de ausência (MCAR/MNAR), valores impossíveis pela definição do atributo.
3. **Classe-alvo e desbalanceamento**: binário, 5,8%, por que acurácia engana, importância do limiar.
4. **Atributos categóricos**: taxa de anomalia por `flag`/`protocol_type`, cauda de `service`.
5. **Numéricos e outliers**: assimetria, log1p, o que os outliers são.
6. **Redundância de atributos**: 5 cópias exatas e correlação entre os 41.
7. **Redundância e inconsistência de instâncias**: cópias, rótulos contraditórios, quantos conflitos parecem ataques reais.
8. **Investigação de vazamento (Seção 7 do enunciado)**: associação, natureza de cada coluna suspeita, `feature_set_gain`.
9. **Protocolo de validação**: CV comum × agrupada; pontuação em todas as linhas (como a métrica oficial), com o valor sem conflitos só como contraste.
10. **Diagnóstico e plano para a Etapa 2**: tabela "problema → evidência → técnica → por quê".

## Qualidade
- Cada gráfico com título, eixos e 1–2 frases de interpretação logo abaixo (a professora avalia o *diagnóstico*, não a quantidade de gráficos).
- **Escreva a conclusão depois de ver o número.** Fatos do NSL-KDD original podem não valer aqui (ex.: outliers não concentram ataques nesta base).
- Salvar figuras relevantes em `results/figures/eda_*.png` (servem ao relatório).
- `python -m nslkdd.audit notebooks/01_eda.ipynb`: zero ERRO. Os ALERTAS das colunas de vazamento são a demonstração do vazamento e já estão justificados no fim do notebook.
- Use os termos do material da disciplina (balanceamento, dados redundantes/inconsistentes, conversão simbólico-numérica); para citar, `Grep` em `docs/_txt/material/`.
