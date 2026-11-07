# Estrutura do Projeto — Detecção de Intrusão (NSL-KDD, base atualizada)

Este documento explica como o projeto está organizado, o que cada arquivo faz e como as peças se encaixam ao longo das 4 etapas do trabalho.

Em 31/10/2026 a base foi trocada por `data/new-database/`: o problema passou a ser **binário** (`normal`/`anomaly`), o teste não tem rótulos e a base traz ausentes, valores impossíveis, duplicatas com rótulos contraditórios e colunas de vazamento. Em 07/11/2026 o enunciado foi atualizado para a base nova (Seção 3.1: formato e nome do arquivo de previsões). Ver a Seção 8.

---

## 1. Visão geral

```
IA-CopaLibertadores/
├── CLAUDE.md                     Regras do projeto carregadas pelo Claude Code em toda sessão
├── README.md                     Como reproduzir o ambiente e executar o trabalho
├── pyproject.toml                Torna src/nslkdd um pacote instalável (pip install -e .)
├── requirements.txt              Dependências (versões mínimas)
├── requirements-lock.txt         Versões exatas instaladas (reprodutibilidade)
├── .gitignore                    Arquivos fora do versionamento (.venv, dados, modelos)
│
├── .claude/
│   ├── settings.json             Permissões do Claude Code para o projeto
│   └── skills/                   8 skills especializadas (ver Seção 3)
│
├── data/
│   ├── new-database/             treino.csv, teste_sem_rotulo.csv, modelo_de_submissao.csv
│   ├── raw/                      NSL-KDD original: fora de uso (contém os rótulos do teste)
│   ├── interim/                  Dados intermediários (se necessário)
│   └── processed/                Dados processados (se necessário)
│
├── docs/
│   ├── ESTRUTURA_DO_PROJETO.md   Este documento
│   ├── enunciado/                PDF do enunciado do trabalho
│   ├── material/                 Slides da disciplina, organizados por tema
│   │   ├── 01-pre-processamento/
│   │   ├── 02-arvore-de-decisao/
│   │   ├── 03-ensemble-random-forest/
│   │   └── 04-regras-de-associacao/
│   └── _txt/                     Versão em texto de todos os PDFs (para busca rápida)
│
├── src/nslkdd/                   Pacote Python com os utilitários do trabalho (ver Seção 4)
├── notebooks/
│   ├── src/                      Fontes dos notebooks em formato .py (# %%)
│   └── *.ipynb                   Notebooks executados (entregáveis)
├── results/
│   ├── experiments.csv           Tabela comparativa acumulada (gerada automaticamente)
│   ├── melhores_configuracoes.json  Melhor configuração por família (rodar_experimentos.py)
│   ├── final_config.json         Configuração final congelada (gerada na Etapa 3)
│   ├── submissao_<N>.csv         Arquivo entregue para o ranking (gerado na Etapa 4; N = TEAM_NUMBER)
│   ├── figures/                  Gráficos para notebooks e relatório
│   └── models/                   Modelo final treinado (.joblib)
├── reports/
│   └── declaracao_ia.md          Registro do uso de IA generativa (exigido pelo enunciado)
├── tests/                        Testes automatizados de dados e de vazamento
├── tools/
│   ├── py2nb.py                  Converte fontes .py em notebooks .ipynb e os executa
│   └── rodar_experimentos.py     Busca por coordenadas e comparação de algoritmos
└── .venv/                        Ambiente virtual Python isolado
```

### O que mudou em relação à organização original

| Antes | Depois | Motivo |
|---|---|---|
| `Enunciado_Trabalho_Competição_IA.pdf` (raiz) | `docs/enunciado/Enunciado_Trabalho_Competicao_IA.pdf` | Documentação separada do código; nome sem acento |
| `material/Etapas-De-Processamento` | `docs/material/01-pre-processamento` | Ordem numérica e nomes padronizados |
| `material/Arvore-De-Decisao` | `docs/material/02-arvore-de-decisao` | Idem |
| `material/Random-Forest` | `docs/material/03-ensemble-random-forest` | O arquivo trata de ensembles em geral |
| `material/Regras-De-Associacao` | `docs/material/04-regras-de-associacao` | Idem |
| `database/` (vazia) | `data/raw/` (com a base baixada) | Convenção padrão de projetos de ciência de dados |
| `.skills/` (vazia) | `.claude/skills/` | Local onde o Claude Code de fato carrega skills do projeto |

---

## 2. Arquivos de configuração da raiz

### `CLAUDE.md`
Memória permanente do projeto, lida automaticamente pelo Claude Code no início de toda sessão. Contém:
- objetivo (binário, F1-macro no teste, submissão por arquivo) e pesos da nota;
- tabela de prazos com a skill correspondente a cada etapa;
- as **8 regras inegociáveis** (teste usado uma única vez, split fixo, tudo dentro do pipeline, só os 41 atributos de rede como preditores, teste nunca alterado, seleção por CV agrupada, justificativa registrada, `data/raw/` fora de uso);
- comandos do ambiente e lembrete de integridade acadêmica.

### `README.md`
Guia para qualquer pessoa (incluindo a professora) reproduzir o trabalho: criação do ambiente, origem dos dados, ordem de execução dos notebooks, verificações e protocolo anti-vazamento resumido.

### `pyproject.toml`
Declara `src/nslkdd` como pacote. Com `pip install -e .`, qualquer notebook faz `import nslkdd` sem manipular `sys.path`. Também configura o `pytest`.

### `requirements.txt` e `requirements-lock.txt`
O primeiro lista as dependências com versões mínimas (numpy, pandas, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib, seaborn, jupyter etc.). O segundo é o `pip freeze` do ambiente instalado, garantindo versões idênticas na reprodução.

### `.gitignore`
Exclui do versionamento o ambiente virtual, caches, dados brutos/processados e modelos treinados.

### `.claude/settings.json`
Libera sem confirmação os comandos rotineiros (testes, auditoria, conversão de notebooks, leitura de `docs/` e `results/`) e **bloqueia edição de `data/raw/`**. A proteção equivalente para `data/new-database/` ainda precisa ser adicionada manualmente.

---

## 3. Skills (`.claude/skills/`)

Skills são guias especializados que o Claude Code carrega sob demanda, apenas quando a tarefa corresponde à descrição da skill. Isso mantém o contexto enxuto e garante que cada etapa siga o mesmo padrão de qualidade.

| Skill | Quando é usada | Conteúdo principal |
|---|---|---|
| `nslkdd-base` | Ler dados, citar números, discutir atributos | API de carga, números verificados, ausentes, valores impossíveis, duplicatas, conflitos, vazamento. Inclui `references/atributos.md` (dicionário dos 41 atributos e pares redundantes) |
| `anti-vazamento` | Qualquer código com fit/transform/reamostragem/teste | Os 11 vazamentos possíveis e a defesa de cada um, critério para colunas suspeitas, padrões certos e proibidos, uso da auditoria, texto-base para o relatório |
| `etapa1-eda` | Notebook 01 | Roteiro em 11 seções, política sobre o teste, cuidados de qualidade |
| `etapa2-preprocessamento` | Notebook 02 (07/11) | Eixos de comparação mapeados aos requisitos 1–5, protocolo de busca, resultados de referência, justificativas citando o material |
| `etapa3-modelagem` | Notebook 03 (21/11) | Algoritmos, busca de hiperparâmetros, formato da tabela comparativa, onde está o ganho real, congelamento do modelo final |
| `etapa4-avaliacao-final` | Notebook 04 (28/11) | Protocolo da submissão única, estimativa por OOF (o teste não tem rótulos), explicabilidade (importâncias, SHAP, regras) |
| `notebook-padrao` | Criar ou alterar qualquer .ipynb | Fluxo .py → .ipynb executado, cabeçalho obrigatório, regras de escrita, checklist de entrega |
| `relatorio-e-apresentacao` | Relatório e slides | Estrutura com orçamento de páginas, modelo da tabela, declaração de IA, referências, roteiro da apresentação |

**Como usar:** basta pedir a tarefa em linguagem natural (ex.: "crie o notebook da etapa 1"); a skill adequada é ativada automaticamente. Também é possível invocá-la pelo nome, ex.: `/etapa1-eda`.

---

## 4. Pacote `src/nslkdd/`

Código reutilizado por todos os notebooks, evitando duplicação e erros.

### `config.py` — constantes
Caminhos do projeto, semente (`RANDOM_STATE = 42`), os 41 atributos permitidos (`FEATURES`), colunas proibidas (`LEAKY_COLUMNS`), cópias exatas (`DERIVED_COLUMNS`), grupos de atributos (`CATEGORICAL`, `NUMERIC`, `RATES`, `HEAVY_TAILED`), classes e limites de domínio (`DOMAIN_MAX`).

### `data.py` — leitura dos dados
| Função | Retorno |
|---|---|
| `load_train()` | `X` (41 atributos), `y` (`normal`/`anomaly`), `groups` (grupo de duplicatas para a CV) |
| `load_train_frame()` | DataFrame completo com as colunas novas (para EDA e auditoria) |
| `load_test(final_evaluation=True)` | `X` e `ids` do teste; **sem o parâmetro, lança erro** |
| `write_submission(ids, y_pred)` | Valida colunas, ids (sem repetidos, mesma ordem), rótulos e vazios contra o modelo e grava `results/submissao_<N>.csv` |
| `check_submission(path)` | Relê o CSV gravado, valida o formato da Seção 3.1 e devolve o SHA-256 (prova de reprodutibilidade) |
| `duplicate_groups`, `conflict_mask`, `evaluation_mask` | Grupos de cópias, linhas contraditórias e linhas usadas para pontuar |

A carga padroniza caixa e espaços dos categóricos, regra fixa por linha que não remove nada.

### `preprocessing.py` — pipelines
`build_pipeline(modelo, **opções)` monta um `Pipeline` do imbalanced-learn na ordem: redundância → domínio → imputação/codificação/transformação → redução → balanceamento → modelo. Cada opção é um eixo de comparação exigido pelo enunciado:

| Parâmetro | Valores | Requisito do enunciado |
|---|---|---|
| `redundancy` | none, drop_duplicates, drop_inconsistent, majority_label, relabel_model | 1 — redundantes/inconsistentes |
| `domain` | nan, clip, none | 1 — valores inconsistentes |
| `imputation` | none (NaN nativo), mean, median, median_indicator, iterative | dados ausentes |
| `balancing` (+ `min_count`, `max_count`) | none, class_weight, ros, rus, smote, borderline_smote, adasyn, smote_tomek, smote_enn, hybrid, tomek, enn | 2 — balanceamento (e seleção de instâncias) |
| `service_encoding`, `categorical_encoding` | onehot, onehot_rare, frequency, target, ordinal | 3 — conversão simbólico-numérica |
| `scaling`, `log_transform` | standard, minmax, robust, quantile, none / True, False | 4 — normalização/transformação |
| `reduction` (+ `reduction_k`) | none, pca, kbest_f, kbest_mi, tree_select | 5 — redução de dimensionalidade |

Como tudo está dentro do pipeline, a validação cruzada ajusta cada etapa **só nos folds de treino**, eliminando vazamento por construção.

### `evaluation.py` — validação e registro
- `cross_validate_pipeline`: CV estratificada e **agrupada por duplicata**, pontuada só em linhas únicas e sem conflito, com o limiar de cada fold aprendido nos demais. `f1_macro_noisy` registra o valor com tudo incluído, como contraste.
- `ExperimentLog`: executa a CV e grava em `results/experiments.csv`. Configurações já executadas são puladas (o cache considera também os hiperparâmetros e a versão dos dados). `leaderboard()` devolve a tabela comparativa ordenada.
- `validation_report(y_true, y_pred)`: métricas e matriz de confusão das previsões OOF. O teste não tem rótulos, então não há relatório de teste.

### `decision.py` — ajuste para F1-macro
`ThresholdClassifier` aprende o limiar de decisão com probabilidades *out-of-fold* do treino, e `cross_val_proba` produz essas previsões sem separar cópias. Usa apenas o treino, portanto é legítimo.

### `models.py` — catálogo e adaptadores
`make_model(nome)` cria cada algoritmo com a semente do projeto. `EncodedLabelClassifier` permite usar o XGBoost (que exige rótulos numéricos) com os rótulos textuais.

### `audit.py` — auditoria de vazamento
- `association_with_target(df, y)`: associação de cada atributo com a classe (eta, V de Cramér, informação mútua), atendendo à Seção 7 do enunciado.
- `feature_set_gain(frame, y, groups, conjuntos)`: quanto cada conjunto de colunas rende em CV; é o teste mais direto para colunas suspeitas.
- `python -m nslkdd.audit notebooks/*.ipynb`: varredura estática que aponta **ERRO** (ex.: `fit` no teste, concatenação treino+teste, uso da base original) e **ALERTA** (ex.: reamostragem fora do pipeline, colunas de vazamento citadas, CV sem grupos, uso do teste fora do notebook 04).

### `explain.py` — explicabilidade
| Função | O que produz |
|---|---|
| `model_importances` | Importância nativa do modelo (árvores/ensembles) |
| `permutation_importances` | Importância por permutação nos atributos originais, medida em F1-macro |
| `shap_values` | Valores SHAP (global e local) |
| `surrogate_rules` | Árvore rasa que imita o modelo: regras legíveis e fidelidade |

---

## 5. Demais pastas

### `notebooks/`
Os entregáveis serão `01_eda.ipynb`, `02_preprocessamento.ipynb`, `03_modelagem.ipynb` e `04_avaliacao_final.ipynb`. Cada um é escrito como `notebooks/src/NN_nome.py` (células marcadas com `# %%`) e convertido com:

```bash
.venv/Scripts/python tools/py2nb.py notebooks/src/01_eda.py notebooks/01_eda.ipynb --execute
```

Vantagens: edição mais rápida e segura que JSON, diffs legíveis e garantia de execução completa num kernel limpo.

### `results/`
- `experiments.csv`: cresce a cada experimento e alimenta a tabela comparativa (20% da nota).
- `final_config.json`: criado no fim da Etapa 3; "congela" o modelo antes de olhar o teste.
- `figures/`, `models/`: gráficos e modelo final.

### `reports/`
- `declaracao_ia.md`: tabela com data, ferramenta, finalidade e revisor de cada uso de IA. Deve ser atualizada ao longo do trabalho e copiada para o relatório.
- O relatório (`relatorio.md`) e a apresentação serão criados aqui na Etapa 4.

### `tests/`
| Arquivo | Verifica |
|---|---|
| `test_data.py` | Tamanhos, padronização do texto, bloqueio do teste, esquema do teste, só os 41 atributos em `X`, duplicatas e conflitos, derivadas exatas, validação da submissão |
| `test_leakage.py` | Samplers não alteram a predição, codificações e imputações, regras de domínio, redundância, CV nunca separa cópias, limiar, cache, modelo salvo, explicabilidade, colunas de vazamento inalcançáveis, scanner |

Executar com `.venv/Scripts/python -m pytest`.

### `docs/_txt/`
Texto extraído de todos os PDFs (enunciado e slides). Permite localizar rapidamente trechos do material da professora para citar nas justificativas.

---

## 6. Fluxo de trabalho por etapa

```
Etapa 1 (01/11)  EDA no treino ─────────────► notebooks/01_eda.ipynb
                                               diagnóstico + plano da Etapa 2
Etapa 2 (07/11)  build_pipeline + ExperimentLog ► notebooks/02_preprocessamento.ipynb
                 (um eixo de pré-processamento por vez, CV no treino)
Etapa 3 (21/11)  ≥3 algoritmos + tuning por CV ► notebooks/03_modelagem.ipynb
                 escolha do modelo ───────────► results/final_config.json (congelado)
Etapa 4 (28/11)  load_test(final_evaluation=True) uma única vez
                 submissão ───────────────────► results/submissao_<N>.csv
                 estimativa OOF + explicabilidade ► notebooks/04_avaliacao_final.ipynb
                 relatório + apresentação ────► reports/
```

Antes de cada entrega:

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m nslkdd.audit notebooks/*.ipynb
```

---

## 7. Garantias contra vazamento de dados (resumo)

1. O teste só é carregado com `final_evaluation=True`, apenas no notebook 04.
2. Todo pré-processamento com parâmetros aprendidos está dentro do pipeline.
3. Só os 41 atributos de rede entram no modelo: as 3 colunas de vazamento e as 5 derivadas ficam de fora por construção.
4. A CV é agrupada por duplicata, então cópias não atravessam a divisão treino/validação.
5. Nenhuma linha do teste é removida, alterada ou reordenada.
6. Escolhas de técnica, hiperparâmetros e limiar de decisão usam apenas CV no treino.
7. O modelo final é congelado em `final_config.json` antes da submissão.
8. `data/raw/` não é usada, pois contém os rótulos do teste.
9. Testes automatizados e a auditoria verificam esses pontos continuamente.

---

## 8. O que mudou com a base nova

| Antes (NSL-KDD original) | Agora | Consequência no projeto |
|---|---|---|
| 5 classes | binário (`normal`/`anomaly`) | métricas, `CLASSES`, limiar em vez de pesos por classe |
| KDDTest+ com rótulos | teste sem rótulos, submissão por arquivo | `write_submission`; estimativa por OOF |
| `attack`, `difficulty` | não existem | CV por grupo de ataque removida |
| sem ausentes | 94% das linhas com algum ausente | eixo `imputation`; NaN nativo nas árvores |
| sem valores impossíveis | 8% das linhas | `DomainCleaner` |
| 16 duplicatas | 13.067 cópias e 1.702 conflitos | CV agrupada, `evaluation_mask`, `relabel_model` |
| categorias limpas | grafias variantes | `normalize_text` na carga |
| 41 atributos | + 5 derivadas e 3 de vazamento | `FEATURES` continua com 41; as demais ficam fora |
