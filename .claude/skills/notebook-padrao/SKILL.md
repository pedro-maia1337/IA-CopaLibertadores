---
name: notebook-padrao
description: Como criar, editar, executar e validar os notebooks entregáveis do trabalho (01_eda, 02_preprocessamento, 03_modelagem, 04_avaliacao_final) com reprodutibilidade total — fontes em formato percent .py convertidas por tools/py2nb.py, kernel nslkdd, estrutura de células, comentários e checagens antes da entrega. Use sempre que for gerar ou alterar qualquer .ipynb.
---

# Padrão de notebooks

Critério "Clareza" (10%) e o entregável nº 1 exigem notebook **comentado e reproduzível do início ao fim**.

## Fluxo de trabalho (eficiente — não editar JSON de .ipynb à mão)
1. Escrever/editar a fonte `notebooks/src/NN_nome.py` em formato percent:
   ```python
   # %% [markdown]
   # ## 3. Classe-alvo e desbalanceamento
   # Texto explicativo em markdown (cada linha começa com "# ").

   # %%
   counts = y.value_counts()
   ```
2. Gerar e executar: `.venv/Scripts/python tools/py2nb.py notebooks/src/01_eda.py notebooks/01_eda.ipynb --execute`
   - Falhou? Corrigir a fonte `.py` e repetir. Iterações rápidas: rodar a fonte como script (`.venv/Scripts/python notebooks/src/01_eda.py`) antes de converter.
   - Execuções longas (Etapas 2–3): rodar em background e conferir o resultado depois.
3. Auditar: `.venv/Scripts/python -m nslkdd.audit notebooks/NN_nome.ipynb` (zero ERRO).
4. O `.ipynb` executado (com saídas) é o entregável; a fonte `.py` é a verdade para edições.

## Cabeçalho obrigatório (primeiras células)
- Markdown: título, disciplina, equipe, etapa e data, objetivo do notebook, resumo dos resultados (preencher ao final).
- Código de setup:
  ```python
  import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
  import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, imblearn
  from nslkdd import config, load_train, load_train_frame, build_pipeline, ExperimentLog
  np.random.seed(config.RANDOM_STATE)
  pd.set_option("display.max_columns", 60)
  sns.set_theme(style="whitegrid", context="notebook")
  print({m.__name__: m.__version__ for m in (np, pd, sklearn, imblearn)})
  ```
- Markdown: "Política de dados" — treino para decisões; teste só no notebook 04.

## Regras de escrita
- Seções numeradas (`## 1.`, `### 1.1`) espelhando a skill da etapa; índice no topo.
- Antes de cada bloco de código: *o que* e *por quê* (1–3 frases). Depois de cada resultado: *o que se conclui*. A justificativa vive no markdown, comentários de código só para o não óbvio.
- Português, termos da disciplina; nomes de variáveis em inglês/curto são aceitáveis (`X`, `y`, `pipe`).
- Nenhum caminho absoluto; usar `config.ROOT`, `config.FIGURES` etc. Figuras salvas com `dpi=150`.
- Nada de células "rascunho", outputs gigantes (`df` inteiro) ou warnings poluindo — use `.head()`, `display` de tabelas resumidas.
- Funções reutilizáveis entre notebooks vão para `src/nslkdd/` (com teste em `tests/` se tiverem lógica), não copiadas de notebook em notebook.
- Seeds sempre via `config.RANDOM_STATE`.

## Checklist antes de entregar
- [ ] "Restart & Run All" limpo (`py2nb --execute` faz isso num kernel novo).
- [ ] `python -m nslkdd.audit` sem ERRO; ALERTAS justificados em markdown.
- [ ] `python -m pytest` passando.
- [ ] Tabelas comparativas vêm de `ExperimentLog.leaderboard()` (não digitadas à mão).
- [ ] Conclusões batem com os números exibidos.
- [ ] `requirements-lock.txt` atualizado se instalou algo (`.venv/Scripts/python -m pip freeze > requirements-lock.txt`).
