---
name: etapa4-avaliacao-final
description: Etapa 4 (entrega 28/11) — avaliação única do modelo congelado no KDDTest+ (F1-macro oficial), relatório por classe, matriz de confusão, análise de erros por ataque inédito e explicabilidade do melhor modelo (importância nativa, permutação, SHAP, árvore substituta). Use ao montar notebooks/04_avaliacao_final.ipynb ou ao interpretar/explicar o modelo final.
---

# Etapa 4 — Avaliação final e explicabilidade

Entregável: `notebooks/04_avaliacao_final.ipynb` (completo e reproduzível do início ao fim). Pré-requisito: `results/final_config.json` congelado na Etapa 3. Se não existir, **pare** e volte à Etapa 3.

## Protocolo (nesta ordem)
1. Ler `final_config.json`; montar o pipeline com `build_pipeline(model, **pipeline_params)` (envolver em `WeightedDecisionClassifier` se `decision_weights`).
2. Treinar no **KDDTrain+ completo** (sem CV). Salvar com `joblib.dump(..., config.MODELS / "final.joblib")`.
3. Só então `X_te, y_te, attack_te = load_test(final_evaluation=True)` — uma vez, sem filtrar linhas.
4. `y_pred = model.predict(X_te)`; `final_report(y_te, y_pred, save_as="matriz_confusao_teste.png")` → F1-macro oficial, acurácia balanceada, precision/recall/F1 por classe, matriz absoluta e normalizada.
5. Tabela CV-treino × teste por classe (mostra o *gap* de generalização; esperado e explicável).
6. **Análise de erros por ataque**: `pd.crosstab(attack_te, y_pred)` + recall por ataque; separar ataques conhecidos × inéditos (lista em `nslkdd-base`). Destaques esperados: `snmpgetattack`/`snmpguess` confundidos com normal (54 linhas idênticas a normal), `httptunnel`, `mscan`, `apache2`, `processtable`.
7. Não reprocessar nem retunar depois de ver o teste. Se surgir ideia, ela vai para "trabalhos futuros" no relatório. Se o teste for executado de novo (ex.: bug), registrar no notebook o motivo e manter o mesmo modelo.

## Explicabilidade do melhor modelo (requisito 7)
```python
from nslkdd.explain import model_importances, permutation_importances, shap_values, surrogate_rules
model_importances(model, X_tr).head(20)                  # global, rápida (árvores)
permutation_importances(model, X_tr, y_tr)               # global, atributos ORIGINAIS, métrica f1_macro
sv, Xt = shap_values(model, X_tr, sample=2000)           # TreeExplainer; sv[..., k] para a classe k
rules, fid = surrogate_rules(model, X_tr, max_depth=4)   # regras legíveis + fidelidade
```
- **Global**: barras de importância + `shap.plots.beeswarm(sv[:, :, k])` por classe para R2L e U2R (as que importam). Relacionar com o domínio: DoS ↔ `serror_rate`/`count`/`flag_S0`; Probe ↔ `diff_srv_rate`/`dst_host_*`; R2L ↔ `service` ftp/telnet, `num_failed_logins`, `dst_bytes`; U2R ↔ `root_shell`, `num_file_creations`, `hot`.
- **Local**: `shap.plots.waterfall(sv[i, :, k])` para 1 acerto e 1 erro de R2L/U2R do teste (explicar *por que* errou é ótimo material de discussão).
- **Regras**: árvore substituta (profundidade 3–4) com fidelidade reportada; para limiares legíveis, explicar que valores estão na escala transformada (log1p + padronização) ou treinar a substituta sobre atributos originais.
- Por padrão use amostras do treino para SHAP/permutação (explicar o modelo); explicações sobre amostras do teste são permitidas por serem só descritivas (não alteram o modelo).
- Se o modelo final for PCA-based, a explicabilidade fica em componentes — por isso a Etapa 2 recomenda seleção de atributos em vez de PCA.

## Discussão dos resultados (vai para o relatório e a apresentação)
- Posição esperada em relação à CV e por quê (mudança de distribuição, ataques inéditos, R2L 0,8%→12,2%).
- Quais escolhas de pré-processamento mais contribuíram (puxar da tabela da Etapa 2/3).
- Limitações: rótulos inconsistentes no teste, U2R com 52 exemplos, SMOTE em dummies, custo computacional.
- Trabalhos futuros (ideias surgidas após ver o teste entram aqui, não no modelo).
