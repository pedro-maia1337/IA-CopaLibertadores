# Trabalho em Competição — Detecção de Intrusão (NSL-KDD)

Disciplina de Inteligência Artificial (Profa. Cristiane Neri Nobre). Enunciado completo:
`docs/enunciado/Enunciado_Trabalho_Competicao_IA.pdf` (texto pesquisável em `docs/_txt/`).

## Objetivo e nota
- Classificação em **5 classes** (`normal`, `DoS`, `Probe`, `R2L`, `U2R`), métrica oficial **F1-macro no KDDTest+**.
- Pesos: desempenho no ranking 40% · qualidade metodológica (sem vazamento) 30% · tabela comparativa 20% · clareza 10%.
- Vazamento derruba fortemente os 30% metodológicos, independentemente do ranking.

## Prazos (2026)
| Etapa | Entrega | Data | Skill |
|---|---|---|---|
| 1. Análise exploratória | `notebooks/01_eda.ipynb` | 24/10 | `etapa1-eda` |
| 2. Pré-processamento | `notebooks/02_preprocessamento.ipynb` | 07/11 | `etapa2-preprocessamento` |
| 3. Modelos + CV | `notebooks/03_modelagem.ipynb` | 21/11 | `etapa3-modelagem` |
| 4. Teste final + XAI + relatório | `notebooks/04_avaliacao_final.ipynb`, `reports/` | 28/11 | `etapa4-avaliacao-final`, `relatorio-e-apresentacao` |

Skills transversais: `nslkdd-base` (fatos da base), `anti-vazamento` (sempre que houver fit/transform/reamostragem/teste), `notebook-padrao` (como criar e executar notebooks).

## Regras inegociáveis
1. **KDDTest+ só no notebook 04, uma única vez**, depois de congelar a configuração em `results/final_config.json`. `nslkdd.load_test()` exige `final_evaluation=True`.
2. Split fixo: nunca concatenar treino+teste, nunca `train_test_split` sobre o teste, nunca re-embaralhar a divisão oficial.
3. Todo passo que aprende algo (codificação, escala, seleção, PCA, balanceamento, remoção de duplicatas) vai **dentro** de `build_pipeline(...)` (imblearn `Pipeline`), para ser ajustado só nos folds de treino da CV.
4. `difficulty` nunca é atributo preditivo (metadado do rótulo). `attack` só é usado para mapear a classe e como grupo de CV.
5. Nunca remover/alterar linhas do KDDTest+ (duplicatas, inconsistências ou outliers do teste ficam como estão).
6. Seleção de modelo e hiperparâmetros por **F1-macro de CV no treino** (`ExperimentLog`). Acurácia nunca é critério principal.
7. Toda decisão de pré-processamento precisa de justificativa técnica registrada (markdown no notebook) — ela vira texto do relatório.

## Ambiente
- Python: `.venv/Scripts/python` (Windows). Kernel Jupyter: `nslkdd`. Pacote local editável: `import nslkdd`.
- Testes: `.venv/Scripts/python -m pytest`. Auditoria: `.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb`.
- Notebooks são escritos como `notebooks/src/NN_nome.py` (formato `# %%`) e gerados com `tools/py2nb.py --execute`.
- Semente global `config.RANDOM_STATE = 42`. Resultados de CV acumulam em `results/experiments.csv` (cache por hash).

## Estrutura
```
data/raw/            KDDTrain+.txt, KDDTest+.txt (não editar)
docs/enunciado/      PDF do enunciado          docs/material/  slides da disciplina
docs/_txt/           texto extraído (use Grep para citar o material da professora)
src/nslkdd/          config, data, preprocessing, evaluation, decision, audit, explain
notebooks/src/       fontes .py dos notebooks  notebooks/  .ipynb executados (entregáveis)
results/             experiments.csv, final_config.json, figures/, models/
reports/             relatório (3–5 págs) e apresentação
tests/               testes de dados e de vazamento
```

## Integridade acadêmica
O enunciado exige que o código seja autoral da equipe e que o uso de IA generativa seja **declarado no relatório** (ferramenta + finalidade). Manter `reports/declaracao_ia.md` atualizado a cada uso relevante.
Escrever notebooks, comentários e relatório em português.
