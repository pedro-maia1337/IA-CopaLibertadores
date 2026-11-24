---
name: anti-vazamento
description: Regras, checklist e auditoria automática contra data leakage no trabalho de detecção de intrusão. Use SEMPRE antes de escrever ou revisar código com fit, fit_transform, fit_resample, scaler, encoder, imputer, PCA, SelectKBest, SMOTE, cross_val, GridSearch, limiar de decisão, remoção de duplicatas/outliers, colunas novas da base, ou qualquer uso do teste; e ao redigir a seção de metodologia do relatório. Vale 30% da nota.
---

# Anti-vazamento (data leakage)

O enunciado reduz fortemente a nota metodológica de quem vazar, mesmo que fique bem no ranking. Trate esta skill como lei.

## Os vazamentos possíveis neste trabalho e a defesa correspondente
| # | Vazamento | Defesa no projeto |
|---|---|---|
| 1 | **Colunas de vazamento** `score_ids_legado`, `tipo_alerta_soc`, `tempo_resposta_analista_min` como preditores | `config.FEATURES` não as contém; `split_xy` e `ColumnTransformer(remainder="drop")` as impedem de chegar ao modelo |
| 2 | Ajustar imputação/scaler/encoder/PCA/seleção no treino+teste ou antes da CV | Tudo dentro de `build_pipeline`; a CV recebe o pipeline inteiro |
| 3 | Balancear (SMOTE/RUS) antes de dividir/antes da CV → amostras sintéticas derivadas do fold de validação | Balanceador é passo do imblearn `Pipeline` (só roda no `fit`) |
| 4 | **Cópias entre folds:** 19% das linhas são cópias; em CV comum, a original valida e a cópia treina | `StratifiedGroupKFold` com `groups` de `load_train()`; a CV do projeto é sempre agrupada |
| 5 | Usar o teste para escolher técnica, hiperparâmetro, limiar ou prior de classe | `load_test` bloqueado sem `final_evaluation=True`; limiar via OOF do treino (`nslkdd.decision`) |
| 6 | Escolher o limiar no mesmo fold que pontua | `_score_folds` usa, em cada fold, um limiar aprendido nos demais |
| 7 | Gerar várias submissões e ficar com a melhor | Pré-registrar `results/final_config.json` no notebook 03; o 04 só lê e submete uma vez |
| 8 | Target encoding de `service` sem cross-fitting | `TargetEncoder` do sklearn (cross-fitting interno) dentro do pipeline |
| 9 | Remover ou reordenar linhas do teste, ou usar o teste para estimar imputação/estatísticas | Limpeza só no treino; o teste passa inteiro pelo pipeline já ajustado |
| 10 | **Recuperar rótulos do teste em `data/raw/`** (a base original contém essas linhas) | Proibido e detectado pelo scanner (`KDDTrain`/`KDDTest` = ERRO). Usar como treino extra também vaza |
| 11 | Pseudo-rótulos, validação adversária ou imputação com o teste | Proibidos: "escolher o melhor modelo apenas por validação cruzada (ou split de validação) dentro do treino" (enunciado, Seção 3.1) |

Limpeza fora do pipeline: só o que é regra fixa por linha, sem aprender nada dos dados, como padronizar texto (`normalize_text`, na carga). Remoção de duplicatas e conflitos usa o rótulo e vai em `redundancy=...`, por fold.

`DomainCleaner` é por linha e sem estado, mas está no pipeline por ser um eixo de comparação, e isso é o mais simples: treino, CV e teste passam pelo mesmo caminho.

## Padrão correto (copiar)
```python
from nslkdd import load_train, ExperimentLog
from nslkdd.models import make_model
X, y, groups = load_train()
log = ExperimentLog()
log.run("etapa2", "RF", make_model("RF"), X, y, groups,
        params={"redundancy": "drop_inconsistent", "balancing": "smote", "min_count": 10000})
```
Padrões proibidos: `scaler.fit(X_test)`, `SMOTE().fit_resample(X, y)` seguido de CV, `pd.concat([train, test])`, `cross_val_score`/`StratifiedKFold` sem grupos, `X = df.drop(columns="class")` (traz as colunas de vazamento), `GridSearchCV(...).fit(X_test, ...)`.

Para busca de hiperparâmetros use `ExperimentLog.run` em laço (ver `tools/rodar_experimentos.py`): `GridSearchCV` só respeita grupos com `.fit(X, y, groups=groups)`, e não aplica o limiar cruzado (cada fold pontuado com o limiar aprendido nos outros).

## Critério para as colunas novas (por que `score_ids_legado` não é legítimo)
Associação alta sozinha não prova vazamento; a prova é a natureza da coluna. Sempre que a base trouxer um atributo suspeito, aplique:
1. **Disponibilidade:** existe no momento de classificar a conexão? `tempo_resposta_analista_min` e `tipo_alerta_soc` só existem depois do evento.
2. **Independência do tráfego:** `audit.feature_set_gain` com o conjunto "só as colunas suspeitas". Se alcança o desempenho do modelo completo sem nenhum dado de rede, não descreve a conexão. Aqui (F1-macro em todas as linhas): 0,889 contra 0,875 dos 41 atributos.
3. **Relação com o fenômeno:** um IDS calcula o score do tráfego. Entre conexões da mesma classe, score alto deveria implicar tráfego diferente; aqui, a probabilidade vista pelo tráfego é igual. Logo é ruído + rótulo.

## Auditoria (rodar antes de cada entrega)
```bash
.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb notebooks/src/*.py
.venv/Scripts/python -m pytest
```
- `[ERRO]` deve ser zero. Cada `[ALERTA]` precisa ser inspecionado e, se legítimo, justificado em markdown ao lado do código.
- Associação atributo×classe (Seção 7 do enunciado), só no treino:
```python
from nslkdd.audit import association_with_target, feature_set_gain
association_with_target(df[config.FEATURES + config.LEAKY_COLUMNS], y)
```
Nenhum dos 41 atributos de rede passa de 0,57; `tipo_alerta_soc` (0,73) e `score_ids_legado` (0,57) lideram.

## Texto pronto para o relatório (adaptar)
"Todas as transformações com parâmetros aprendidos (imputação, codificação, normalização, seleção/redução, balanceamento e remoção de redundâncias) foram encapsuladas em um `Pipeline` do imbalanced-learn e ajustadas exclusivamente nos folds de treino de uma validação cruzada estratificada e agrupada por duplicata (k=5): cópias de uma linha nunca ficam em lados opostos da divisão. Os atributos `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min` foram removidos por serem posteriores ao evento e por se comportarem como cópias ruidosas do rótulo: sozinhos, sem nenhum dado de rede, atingem F1-macro de 0,992. Eles também estão no conjunto de teste, mas não estariam disponíveis em operação. O limiar de decisão foi ajustado com previsões out-of-fold do treino. O teste foi carregado uma única vez, após o congelamento da configuração final, sem qualquer remoção ou alteração de instâncias."
