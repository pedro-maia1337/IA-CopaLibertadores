---
name: etapa4-avaliacao-final
description: Etapa 4 (entrega 28/11) — gerar uma única vez a submissão do modelo congelado para teste_sem_rotulo.csv, estimar o desempenho por OOF no treino (o teste não tem rótulos) e fazer a explicabilidade do melhor modelo (importância nativa, permutação, SHAP, árvore substituta). Use ao montar notebooks/04_avaliacao_final.ipynb ou ao interpretar/explicar o modelo final.
---

# Etapa 4 — Submissão final e explicabilidade

Entregáveis: `notebooks/04_avaliacao_final.ipynb` (completo e reproduzível do início ao fim) e `results/submissao_<N>.csv` (`N` = `config.TEAM_NUMBER`; se for `None`, pergunte o número do grupo no Canvas). Pré-requisito: `results/final_config.json` congelado na Etapa 3. Se não existir, **pare** e volte à Etapa 3.

**O teste não tem rótulos.** Quem calcula o F1-macro é a professora. Não existe matriz de confusão no teste nem análise de erros no teste: toda métrica deste notebook vem de previsões out-of-fold do treino, e deve ser apresentada como **estimativa**.

## Protocolo (nesta ordem)
1. Ler `final_config.json`; montar o pipeline com `build_pipeline(model, **pipeline_params)` e envolver em `ThresholdClassifier` (aprende o limiar via OOF agrupado no treino).
2. Treinar no **`treino.csv` completo**. Salvar com `joblib.dump(..., config.MODELS / "final.joblib")`.
3. Estimativa honesta: `cross_val_proba` + limiar + `validation_report` nas linhas de `evaluation_mask`. Reporte precisão, recall e F1 **das duas classes** (é o que a professora divulga) e o F1-macro. Mostrar também `f1_macro_noisy`, explicando por que ele não estima o teste (conflitos existem só por construção no treino).
4. Só então `X_te, ids = load_test(final_evaluation=True)`, uma vez, sem filtrar nem reordenar linhas.
5. `y_pred = model.predict(X_te)`; `write_submission(ids, y_pred)` valida colunas, ids, ordem, rótulos em minúsculas e vazios e grava `results/submissao_<N>.csv` (fim de linha `
`, bytes determinísticos).
6. Verificações sem usar rótulos: `print(check_submission())` (SHA-256 do arquivo; o mesmo hash vai para o relatório e deve bater com o arquivo enviado ao Canvas), 24.333 linhas, sem nulos, proporção prevista de `anomaly` (só para detectar erro grosseiro; **não** ajustar o limiar por ela).
7. **Reprodutibilidade (Seção 3.1):** o notebook entregue precisa gerar exatamente o arquivo enviado. Reexecute o notebook do zero antes do envio e confira o hash. Nunca edite o CSV à mão.
8. Não reprocessar nem retunar depois de gerar a submissão. Se surgir ideia, ela vai para "trabalhos futuros". Se a submissão precisar ser refeita (ex.: bug), registrar o motivo no notebook e manter o mesmo modelo.

## Explicabilidade do melhor modelo (requisito 7)
```python
from nslkdd.explain import model_importances, permutation_importances, shap_values, surrogate_rules
model_importances(model, X_tr).head(20)                  # nativa; HGB não tem, use permutação
permutation_importances(model, X_tr, y_tr)               # atributos ORIGINAIS, métrica f1_macro
sv, Xt = shap_values(model, X_tr, sample=2000)           # TreeExplainer (XGB/RF)
rules, fid = surrogate_rules(model, X_tr, max_depth=4)   # regras legíveis + fidelidade
```
- Se o modelo final for um voto de vários modelos, explique o membro mais forte ou use a permutação sobre o ensemble inteiro.
- **Global**: barras de importância + `shap.plots.beeswarm` da classe `anomaly`. Relacionar com o domínio: `flag`, `serror_rate`, `same_srv_rate` e `service` costumam dominar.
- **Local**: `shap.plots.waterfall` para 1 acerto e 1 erro da validação OOF.
- **Regras**: árvore substituta (profundidade 3–4) com fidelidade reportada.
- **Prova de que o vazamento saiu:** nenhuma das colunas de `config.LEAKY_COLUMNS` pode aparecer entre os atributos explicados. Mostre isso, é um ótimo argumento metodológico.
- Use amostras do treino para SHAP/permutação. Não há rótulos no teste, e explicar o modelo não exige o teste.

## Discussão dos resultados (vai para o relatório e a apresentação)
- Por que o F1 bruto parou em ~0,88 e o F1 sobre linhas limpas é a medida correta.
- Custo de abrir mão das colunas de vazamento (quantificado na EDA), e por que isso é o correto: um detector que depende da triagem do SOC detecta o que já foi detectado.
- Quais escolhas de pré-processamento mais contribuíram (puxar da tabela da Etapa 2/3).
- Limitações: o rótulo verdadeiro dos conflitos é desconhecido, o ruído pode existir também onde não há cópia para denunciá-lo, SMOTE em dummies, ~40% de ausentes em `src_bytes`.
- Trabalhos futuros (ideias surgidas após a submissão entram aqui, não no modelo).
