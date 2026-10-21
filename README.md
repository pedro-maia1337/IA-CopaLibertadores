# Detecção de Intrusão em Redes — NSL-KDD

Trabalho em competição da disciplina de Inteligência Artificial (PUC Minas, Profa. Cristiane Neri Nobre).
Classificação em 5 classes (normal, DoS, Probe, R2L, U2R); métrica oficial: **F1-macro no KDDTest+**.

## Reproduzir
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # ou requirements-lock.txt (versões exatas)
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m ipykernel install --user --name nslkdd --display-name "Python (nslkdd)"
```
Dados: `KDDTrain+.txt` e `KDDTest+.txt` em `data/raw/` (Kaggle `hassan06/nslkdd`; 125.973 e 22.544 linhas).

Executar os notebooks na ordem `notebooks/01_eda` → `02_preprocessamento` → `03_modelagem` → `04_avaliacao_final`
(ou regerar a partir das fontes: `.venv/Scripts/python tools/py2nb.py notebooks/src/01_eda.py notebooks/01_eda.ipynb --execute`).

Verificações: `.venv/Scripts/python -m pytest` e `.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb`.

## Estrutura
| Pasta | Conteúdo |
|---|---|
| `data/raw/` | base original (somente leitura) |
| `docs/` | enunciado, slides da disciplina e versão em texto (`_txt/`) |
| `src/nslkdd/` | carga, pipelines de pré-processamento, CV/registro de experimentos, auditoria de vazamento, explicabilidade |
| `notebooks/` | entregáveis por etapa (`src/` = fontes .py) |
| `results/` | `experiments.csv` (tabela comparativa), `final_config.json`, figuras, modelos |
| `reports/` | relatório, apresentação, declaração de uso de IA |
| `tests/` | testes de dados e de vazamento |

## Protocolo anti-vazamento
Todo pré-processamento com parâmetros aprendidos vive dentro de um `Pipeline` imbalanced-learn ajustado só nos folds de treino;
o KDDTest+ é carregado uma única vez, no notebook 04, após o congelamento de `results/final_config.json`.
