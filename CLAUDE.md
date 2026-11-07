# Trabalho em Competição — Detecção de Intrusão (NSL-KDD, base atualizada)

Disciplina de Inteligência Artificial (Profa. Cristiane Neri Nobre). Enunciado completo:
`docs/enunciado/Enunciado_Trabalho_Competicao_IA.pdf` (texto pesquisável em `docs/_txt/`).

A base foi trocada em 31/10/2026 e o enunciado foi atualizado em 07/11/2026 para refletir a troca: os dados são
`data/new-database/treino.csv`, `teste_sem_rotulo.csv` e `modelo_de_submissao.csv` (distribuídos pelo Canvas).

## Objetivo e nota
- Classificação **binária** (`normal`, `anomaly`), métrica oficial **F1-macro no teste**, calculada pela professora.
- O teste não tem rótulos: o entregável é `results/submissao_<N>.csv` (`N` = `config.TEAM_NUMBER`, grupo do Canvas), com `id,class_prevista`,
  cabeçalho, vírgula, rótulos em minúsculas, um id por linha na ordem de `modelo_de_submissao.csv` (enunciado, Seção 3.1). Envio pelo Canvas até 28/11.
- O notebook 04 precisa gerar **exatamente** o arquivo enviado; senão as previsões são desconsideradas. Registrar o SHA-256 de `check_submission()`.
- A professora divulga precisão, recall e F1 por classe; o ranking é pelo F1-macro.
- Pesos: desempenho no ranking 40% · qualidade metodológica (sem vazamento) 30% · tabela comparativa 20% · clareza 10%.
- Vazamento derruba fortemente os 30% metodológicos, independentemente do ranking.

## Prazos (2026)
| Etapa | Entrega | Data | Skill |
|---|---|---|---|
| 1. Análise exploratória | `notebooks/01_eda.ipynb` | 01/11 | `etapa1-eda` |
| 2. Pré-processamento | `notebooks/02_preprocessamento.ipynb` | 07/11 | `etapa2-preprocessamento` |
| 3. Modelos + CV | `notebooks/03_modelagem.ipynb` | 21/11 | `etapa3-modelagem` |
| 4. Submissão + XAI + relatório | `notebooks/04_avaliacao_final.ipynb`, `results/submissao_<N>.csv`, `reports/` | 28/11 | `etapa4-avaliacao-final`, `relatorio-e-apresentacao` |

Skills transversais: `nslkdd-base` (fatos da base), `anti-vazamento` (sempre que houver fit/transform/reamostragem/teste), `notebook-padrao` (como criar e executar notebooks).

## Regras inegociáveis
1. **O teste só entra no notebook 04, uma única vez**, depois de congelar a configuração em `results/final_config.json`. `load_test()` exige `final_evaluation=True`.
2. Split fixo: nunca concatenar treino+teste, nunca usar o teste para ajustar nada (nem pseudo-rótulos, nem validação adversária).
3. Todo passo que aprende algo (imputação, codificação, escala, seleção, PCA, balanceamento, remoção de duplicatas) vai **dentro** de `build_pipeline(...)` (imblearn `Pipeline`), para ser ajustado só nos folds de treino da CV.
4. **Só os 41 atributos de rede (`config.FEATURES`) são preditores.** `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min` são vazamento (cópias ruidosas do rótulo, posteriores ao evento) e nunca entram no modelo, mesmo estando no teste. As 5 derivadas (`src_bytes_kb`, `dst_bytes_kb`, `total_bytes`, `same_srv_pct`, `protocolo_cod`) são cópias exatas e também ficam de fora.
5. Nunca remover, alterar ou reordenar linhas do teste. Só valem transformações por linha, sem parâmetros aprendidos no teste.
6. Seleção de modelo e hiperparâmetros por **F1-macro de CV no treino** (`ExperimentLog`), com CV agrupada por duplicata. Acurácia nunca é critério principal.
7. Toda decisão de pré-processamento precisa de justificativa técnica registrada (markdown no notebook) — ela vira texto do relatório.
8. **`data/raw/` (NSL-KDD original) não pode ser usada:** contém as linhas do teste com os rótulos. Cruzar as bases para recuperar rótulos é fraude; usá-la como treino extra também vaza o teste.

## Ambiente
- Python: `.venv/Scripts/python` (Windows). Kernel Jupyter: `nslkdd`. Pacote local editável: `import nslkdd`.
- Testes: `.venv/Scripts/python -m pytest`. Auditoria: `.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb`.
- Notebooks são escritos como `notebooks/src/NN_nome.py` (formato `# %%`) e gerados com `tools/py2nb.py --execute`.
- Experimentos de referência: `tools/rodar_experimentos.py etapa2|etapa3`.
- Semente global `config.RANDOM_STATE = 42`. Resultados de CV acumulam em `results/experiments.csv` (cache por hash).

## Estrutura
```
data/new-database/   treino.csv, teste_sem_rotulo.csv, modelo_de_submissao.csv (não editar)
data/raw/            NSL-KDD original: fora de uso (regra 8)
docs/enunciado/      PDF do enunciado          docs/material/  slides da disciplina
docs/_txt/           texto extraído (use Grep para citar o material da professora)
src/nslkdd/          config, data, preprocessing, models, evaluation, decision, search, audit, explain
notebooks/src/       fontes .py dos notebooks  notebooks/  .ipynb executados (entregáveis)
results/             experiments.csv, final_config.json, submissao_<N>.csv, figures/, models/
reports/             relatório (3–5 págs) e apresentação
tests/               testes de dados e de vazamento
tools/               py2nb.py, rodar_experimentos.py
```

## Integridade acadêmica
O enunciado exige que o código seja autoral da equipe e que o uso de IA generativa seja **declarado no relatório** (ferramenta + finalidade). Manter `reports/declaracao_ia.md` atualizado a cada uso relevante.
Escrever notebooks, comentários e relatório em português.
