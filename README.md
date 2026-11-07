# Detecção de Intrusão em Redes — NSL-KDD (base da competição)

Trabalho em competição da disciplina de Inteligência Artificial (PUC Minas, Profa. Cristiane Neri Nobre).
Classificação binária (`normal`, `anomaly`); métrica oficial: **F1-macro no conjunto de teste**, calculada pela professora a partir de `results/submissao_<N>.csv` (`N` = número da equipe, `config.TEAM_NUMBER`).

## Reproduzir
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # ou requirements-lock.txt (versões exatas)
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m ipykernel install --user --name nslkdd --display-name "Python (nslkdd)"
```
Dados em `data/new-database/`: `treino.csv` (69.833 linhas, 49 colunas + `class`), `teste_sem_rotulo.csv` (24.333 linhas, `id` + 49 colunas) e `modelo_de_submissao.csv`.
`treino.csv.pdf` é a impressão do próprio CSV e não é usado.

Executar os notebooks na ordem `notebooks/01_eda` → `02_preprocessamento` → `03_modelagem` → `04_avaliacao_final`
(ou regerar a partir das fontes: `.venv/Scripts/python tools/py2nb.py notebooks/src/01_eda.py notebooks/01_eda.ipynb --execute`).

Os experimentos de referência são reproduzíveis com `.venv/Scripts/python tools/rodar_experimentos.py etapa2` (e depois `etapa3`); o resultado fica em `results/experiments.csv`.

Verificações: `.venv/Scripts/python -m pytest` e `.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb`.

## Estrutura
| Pasta | Conteúdo |
|---|---|
| `data/new-database/` | base da competição (somente leitura, fora do git) |
| `data/raw/` | NSL-KDD original: **fora de uso**, pois contém os rótulos do teste |
| `docs/` | enunciado, slides da disciplina e versão em texto (`_txt/`) |
| `src/nslkdd/` | carga, pipelines de pré-processamento, CV/registro de experimentos, limiar de decisão, auditoria de vazamento, explicabilidade |
| `notebooks/` | entregáveis por etapa (`src/` = fontes .py) |
| `results/` | `experiments.csv` (tabela comparativa), `final_config.json`, `submissao_<N>.csv`, figuras, modelos |
| `reports/` | relatório, apresentação, declaração de uso de IA |
| `tests/` | testes de dados e de vazamento |
| `tools/` | conversão de notebooks e experimentos de referência |

## Protocolo anti-vazamento
- Só os 41 atributos de rede são preditores. `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min` são cópias ruidosas do rótulo, posteriores ao evento, e ficam fora do modelo; a evidência está na Seção 8 do notebook 01.
- Todo pré-processamento com parâmetros aprendidos vive dentro de um `Pipeline` imbalanced-learn ajustado só nos folds de treino.
- A CV é agrupada por duplicata: cópias de uma linha nunca ficam em lados opostos da divisão.
- O limiar de decisão vem de previsões out-of-fold do treino.
- O teste é carregado uma única vez, no notebook 04, após o congelamento de `results/final_config.json`, sem remover ou reordenar linhas.
