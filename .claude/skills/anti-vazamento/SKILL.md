---
name: anti-vazamento
description: Regras, checklist e auditoria automática contra data leakage no trabalho NSL-KDD. Use SEMPRE antes de escrever ou revisar código com fit, fit_transform, fit_resample, scaler, encoder, PCA, SelectKBest, SMOTE, cross_val, GridSearch, remoção de duplicatas/outliers, ou qualquer uso do KDDTest+; e ao redigir a seção de metodologia do relatório. Vale 30% da nota.
---

# Anti-vazamento (data leakage)

O enunciado reduz fortemente a nota metodológica de quem vazar, mesmo que fique bem no ranking. Trate esta skill como lei.

## Os 8 vazamentos possíveis neste trabalho e a defesa correspondente
| # | Vazamento | Defesa no projeto |
|---|---|---|
| 1 | Ajustar scaler/encoder/PCA/seleção no treino+teste ou antes da CV | Tudo dentro de `build_pipeline`; CV recebe o pipeline inteiro |
| 2 | Balancear (SMOTE/RUS) antes de dividir/antes da CV → amostras sintéticas derivadas do fold de validação | Balanceador é passo do imblearn `Pipeline` (só roda no `fit`) |
| 3 | Usar o KDDTest+ para escolher técnica, hiperparâmetro, limiar ou pesos de classe | `load_test` bloqueado sem `final_evaluation=True`; pesos de decisão via OOF do treino (`nslkdd.decision`) |
| 4 | Avaliar no teste várias vezes e ficar com a melhor | Pré-registrar `results/final_config.json` no notebook 03; o 04 só lê e avalia uma vez |
| 5 | Atributo `difficulty` como preditor (derivado do rótulo) | `split_xy` nunca o inclui; justificar no relatório |
| 6 | Target encoding de `service` sem cross-fitting | `TargetEncoder` do sklearn (cross-fitting interno) dentro do pipeline |
| 7 | Remover duplicatas/outliers/inconsistências olhando o teste, ou remover linhas do teste | Limpeza só no treino; o teste é avaliado integralmente |
| 8 | Re-split ou embaralhar treino+teste | Proibido pelo enunciado; validação só por CV/holdout **dentro** do KDDTrain+ |

Nota sobre limpeza fora do pipeline: remover duplicatas exatas do treino **antes** da CV é aceitável (não usa o teste e não aprende estatística), mas a forma mais rigorosa é `redundancy=...` em `build_pipeline` (roda por fold). Prefira a forma rigorosa e diga isso no relatório.

## Padrão correto (copiar)
```python
from nslkdd import load_train, build_pipeline, ExperimentLog
X, y, attack = load_train()
log = ExperimentLog()
log.run("etapa2", "RF", RandomForestClassifier(random_state=42),
        X, y, params={"balancing": "smote", "min_count": 2000, "scaling": "standard"})
```
Padrões proibidos: `scaler.fit(X_test)`, `SMOTE().fit_resample(X, y)` seguido de `cross_val_score`, `pd.concat([train, test])`, `train_test_split` sobre a base completa, `GridSearchCV(...).fit(X_test, ...)`.

Para um holdout rápido de iteração, use `train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)` **apenas sobre o KDDTrain+** (nomeie `X_val`, não `X_test`). O scanner acusa todo `train_test_split`; confira o contexto e explique na célula markdown.

## Auditoria (rodar antes de cada entrega)
```bash
.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb notebooks/src/*.py
.venv/Scripts/python -m pytest
```
- `[ERRO]` deve ser zero. Cada `[ALERTA]` precisa ser inspecionado e, se legítimo, justificado em markdown ao lado do código.
- Associação atributo×classe (Seção 7 do enunciado), só no treino:
```python
from nslkdd.audit import association_with_target
association_with_target(load_train_frame()[config.FEATURES + ["difficulty"]], y)
```
Discutir todo atributo com associação muito alta (≥ 0,9): é preditor legítimo do fenômeno (ex.: `same_srv_rate` para DoS) ou proxy do rótulo? No NSL-KDD nenhum dos 41 atributos passa de 0,83; o único removido por vazamento é `difficulty`, por **natureza** (metadado de rotulagem), não por correlação.

## Texto pronto para o relatório (adaptar)
"Todas as transformações com parâmetros aprendidos (codificação, normalização, seleção/redução, balanceamento e remoção de redundâncias) foram encapsuladas em um `Pipeline` do imbalanced-learn e ajustadas exclusivamente nos folds de treino da validação cruzada estratificada (k=5) sobre o KDDTrain+. O KDDTest+ foi carregado uma única vez, após o congelamento da configuração final, sem qualquer remoção ou alteração de instâncias. O atributo `difficulty` foi descartado por ser derivado do processo de rotulagem e indisponível em operação."
