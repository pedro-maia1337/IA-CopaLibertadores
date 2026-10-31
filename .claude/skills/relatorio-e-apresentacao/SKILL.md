---
name: relatorio-e-apresentacao
description: Redação do relatório final (3 a 5 páginas) e da apresentação do trabalho de detecção de intrusão — estrutura, tabela comparativa (pré-processamento × algoritmo × métricas), justificativa técnica de cada escolha citando o material da disciplina, discussão de vazamento e a declaração obrigatória de uso de IA generativa. Use ao escrever, revisar ou encurtar o relatório, montar slides ou preparar a defesa oral.
---

# Relatório (3–5 páginas) e apresentação

Fonte: `reports/relatorio.md` (rascunho em Markdown); exportar para PDF/DOCX no final. Figuras de `results/figures/`, tabelas de `ExperimentLog.leaderboard()`.

## Estrutura e orçamento de páginas (≈ 4 págs + referências)
1. **Introdução** (¼ pág): problema, base, métrica F1-macro e por que não acurácia.
2. **Base e diagnóstico** (½ pág): problema binário com 5,8% de `anomaly`, principais achados da EDA, colunas de vazamento (Seção 7), rótulos contraditórios, ausentes.
3. **Metodologia** (1 pág): protocolo de validação (CV estratificada agrupada por duplicata, pontuada em linhas únicas e sem conflito, com limiar cruzado), pipeline e ordem das etapas (diagrama simples), parágrafo anti-vazamento (texto-base na skill `anti-vazamento`), critério de escolha do modelo pré-declarado.
4. **Pré-processamento: escolhas e justificativas** (1 pág): uma subseção curta por requisito 1–5 + ausentes. Formato fixo: *alternativas testadas → resultado (F1-macro ± dp) → decisão → justificativa técnica + referência ao material*.
5. **Modelos e tabela comparativa** (1 pág): tabela principal (estratégia × algoritmo × F1-macro, F1 anomaly, precisão, recall), discussão de quem ganha e por quê.
6. **Resultado final e explicabilidade** (¾ pág): estimativa OOF, F1-macro oficial divulgado pela professora (se houver), matriz de confusão OOF, importâncias/SHAP, 1 exemplo local.
7. **Conclusão e limitações** (¼ pág).
8. **Declaração de uso de IA generativa** (obrigatória, Seção 8 do enunciado).
9. **Referências**.

## Tabela comparativa — formato
| Redund. | Domínio | Imputação | Codif. `service` | Transf. | Balanceamento | Redução | Algoritmo | F1-macro (CV) | F1 anomaly | Precisão | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
Ordenar por F1-macro; negrito na linha escolhida; ± dp em F1-macro. Se não couber, tabela resumida no corpo e completa em apêndice/notebook (dizer onde).

## Declaração de IA (modelo — manter honesta e específica)
"Ferramentas de IA generativa utilizadas: Claude Code (Anthropic). Finalidades: [ex.: organização do ambiente e estrutura de pastas; adaptação do pacote à base atualizada; apoio à depuração de código; revisão de redação]. Todo o código e texto foram revisados, compreendidos e validados pela equipe, que é responsável pelas decisões metodológicas." Atualizar `reports/declaracao_ia.md` ao longo do trabalho e copiar para o relatório. Não omitir usos relevantes: o enunciado penaliza falta de declaração.

## Referências úteis (verificáveis)
- Tavallaee, M. et al. A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA, 2009.
- Batista, G.; Prati, R.; Monard, M. A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data. SIGKDD Explorations, 6(1), 2004 (arquivo "Balanceamento de dados.pdf" do material).
- Chawla, N. et al. SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 2002.
- Breiman, L. Random Forests. Machine Learning, 45, 2001.
- Chen, T.; Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.
- Lundberg, S.; Lee, S. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.
- Faceli, K. et al. Inteligência Artificial: Uma Abordagem de Aprendizado de Máquina. LTC, 2011 (cap. 3 — citado nos slides de pré-processamento).
Para citar os slides da professora, localizar o trecho com `Grep` em `docs/_txt/material/` e referenciar pelo título do slide.

## Estilo
- Cada afirmação numérica deve existir no notebook. Nada de "melhorou significativamente" sem ± dp.
- Frases curtas, voz ativa, termos da disciplina. Evitar repetir o enunciado.
- Encurtar cortando descrição de código, nunca justificativas ou a tabela.

## Apresentação (≈ 10 min, 8–10 slides)
Problema e métrica → base e armadilhas (colunas de vazamento, rótulos contraditórios, cópias, ausentes) → protocolo anti-vazamento (diagrama do pipeline) → comparação de pré-processamento (1 gráfico por eixo, só os relevantes) → comparação de modelos (tabela) → resultado + matriz de confusão OOF → explicabilidade → lições e limitações. Preparar respostas para: "onde o balanceamento é aplicado?", "por que não usar acurácia?", "por que removeu `score_ids_legado`, se ele também está no teste?", "como escolheu os hiperparâmetros sem usar o teste?", "por que seu F1 de CV é maior que o F1 bruto?".
