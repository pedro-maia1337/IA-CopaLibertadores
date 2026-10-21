# Estrutura do Projeto — Detecção de Intrusão (NSL-KDD)

Este documento explica como o projeto está organizado, o que cada arquivo faz e como as peças se encaixam ao longo das 4 etapas do trabalho.

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
│   ├── raw/                      KDDTrain+.txt e KDDTest+.txt originais (somente leitura)
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
│   ├── final_config.json         Configuração final congelada (gerada na Etapa 3)
│   ├── figures/                  Gráficos para notebooks e relatório
│   └── models/                   Modelo final treinado (.joblib)
├── reports/
│   └── declaracao_ia.md          Registro do uso de IA generativa (exigido pelo enunciado)
├── tests/                        Testes automatizados de dados e de vazamento
├── tools/
│   └── py2nb.py                  Converte fontes .py em notebooks .ipynb e os executa
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
- objetivo (5 classes, F1-macro no KDDTest+) e pesos da nota;
- tabela de prazos com a skill correspondente a cada etapa;
- as **7 regras inegociáveis** (teste usado uma única vez, split fixo, tudo dentro do pipeline, `difficulty` descartado, teste nunca alterado, seleção por CV, justificativa registrada);
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
Libera sem confirmação os comandos rotineiros (testes, auditoria, conversão de notebooks, leitura de `docs/` e `results/`) e **bloqueia edição de `data/raw/`**, protegendo a base original.

---

## 3. Skills (`.claude/skills/`)

Skills são guias especializados que o Claude Code carrega sob demanda, apenas quando a tarefa corresponde à descrição da skill. Isso mantém o contexto enxuto e garante que cada etapa siga o mesmo padrão de qualidade.

| Skill | Quando é usada | Conteúdo principal |
|---|---|---|
| `nslkdd-base` | Ler dados, citar números, discutir atributos | API de carga, tabela de classes treino×teste, problemas conhecidos, 17 ataques inéditos, mudança de distribuição. Inclui `references/atributos.md` (dicionário dos 41 atributos e pares redundantes) |
| `anti-vazamento` | Qualquer código com fit/transform/reamostragem/teste | Os 8 vazamentos possíveis e a defesa de cada um, padrões certos e proibidos, uso da auditoria, texto-base para o relatório |
| `etapa1-eda` | Notebook 01 (24/10) | Roteiro em 11 seções, tabela de diagnóstico problema → evidência → técnica |
| `etapa2-preprocessamento` | Notebook 02 (07/11) | Eixos de comparação mapeados aos requisitos 1–5, protocolo de busca, resultados de referência, justificativas citando o material |
| `etapa3-modelagem` | Notebook 03 (21/11) | Algoritmos e espaços de hiperparâmetros, formato da tabela comparativa, táticas legítimas para subir no ranking, congelamento do modelo final |
| `etapa4-avaliacao-final` | Notebook 04 (28/11) | Protocolo de avaliação única no teste, análise de erros por ataque, explicabilidade (importâncias, SHAP, regras) |
| `notebook-padrao` | Criar ou alterar qualquer .ipynb | Fluxo .py → .ipynb executado, cabeçalho obrigatório, regras de escrita, checklist de entrega |
| `relatorio-e-apresentacao` | Relatório e slides | Estrutura com orçamento de páginas, modelo da tabela, declaração de IA, referências, roteiro da apresentação |

**Como usar:** basta pedir a tarefa em linguagem natural (ex.: "crie o notebook da etapa 1"); a skill adequada é ativada automaticamente. Também é possível invocá-la pelo nome, ex.: `/etapa1-eda`.

---

## 4. Pacote `src/nslkdd/`

Código reutilizado por todos os notebooks, evitando duplicação e erros.

### `config.py` — constantes
Caminhos do projeto, semente (`RANDOM_STATE = 42`), nomes das 41 colunas, grupos de atributos (`CATEGORICAL`, `NUMERIC`, `BINARY`, `HEAVY_TAILED`), colunas proibidas (`LEAKY_COLUMNS = ["difficulty"]`), coluna constante (`num_outbound_cmds`), as 5 classes e o mapeamento completo `ATTACK_TO_CATEGORY` (40 ataques de treino e teste).

### `data.py` — leitura dos dados
| Função | Retorno |
|---|---|
| `load_train()` | `X` (41 atributos), `y` (5 classes), `attack` (nome original do ataque) |
| `load_train_frame()` | DataFrame completo, incluindo `difficulty` e `category` (para EDA) |
| `load_test(final_evaluation=True)` | O mesmo para o KDDTest+; **sem o parâmetro, lança erro** |

Também falha se aparecer um nome de ataque sem categoria mapeada.

### `preprocessing.py` — pipelines
`build_pipeline(modelo, **opções)` monta um `Pipeline` do imbalanced-learn na ordem: redundância → codificação/transformação → redução → balanceamento → modelo. Cada opção é um eixo de comparação exigido pelo enunciado:

| Parâmetro | Valores | Requisito do enunciado |
|---|---|---|
| `redundancy` | none, drop_duplicates, drop_inconsistent, majority_label | 1 — redundantes/inconsistentes |
| `balancing` (+ `min_count`, `max_count`) | none, class_weight, ros, rus, smote, borderline_smote, adasyn, smote_tomek, smote_enn, hybrid, tomek, enn | 2 — balanceamento (e seleção de instâncias) |
| `service_encoding` | onehot, onehot_rare, frequency, target, ordinal | 3 — conversão simbólico-numérica |
| `scaling`, `log_transform` | standard, minmax, robust, quantile, none / True, False | 4 — normalização/transformação |
| `reduction` (+ `reduction_k`) | none, pca, kbest_f, kbest_mi, tree_select | 5 — redução de dimensionalidade |

Como tudo está dentro do pipeline, a validação cruzada ajusta cada etapa **só nos folds de treino**, eliminando vazamento por construção.

### `evaluation.py` — validação e registro
- `make_cv("stratified")`: StratifiedKFold de 5 folds (critério principal).
- `make_cv("attack_group", attack=attack)`: cada tipo de ataque inteiro em um único fold, simulando os ataques inéditos do teste (diagnóstico de generalização).
- `ExperimentLog`: executa a CV, mede F1-macro (média e desvio), acurácia balanceada, acurácia e F1 por classe, e grava em `results/experiments.csv`. Configurações já executadas são puladas (cache). `leaderboard()` devolve a tabela comparativa ordenada.
- `final_report(y_true, y_pred)`: métricas oficiais e matriz de confusão (absoluta e normalizada) para a Etapa 4.

### `decision.py` — ajuste para F1-macro
`WeightedDecisionClassifier` aprende pesos por classe sobre probabilidades *out-of-fold* do treino e desloca a decisão a favor das classes raras (R2L, U2R). Usa apenas o treino, portanto é legítimo.

### `models.py` — adaptadores
`EncodedLabelClassifier` permite usar o XGBoost (que exige rótulos numéricos) com os rótulos textuais do projeto, mantendo compatibilidade com `ExperimentLog` e com as métricas por classe.

### `audit.py` — auditoria de vazamento
- `association_with_target(df, y)`: associação de cada atributo com a classe (eta, V de Cramér, informação mútua), atendendo à Seção 7 do enunciado.
- `python -m nslkdd.audit notebooks/*.ipynb`: varredura estática que aponta **ERRO** (ex.: `fit` no teste, concatenação treino+teste) e **ALERTA** (ex.: reamostragem fora do pipeline, uso de `difficulty`, uso do teste fora do notebook 04).

### `explain.py` — explicabilidade
| Função | O que produz |
|---|---|
| `model_importances` | Importância nativa do modelo (árvores/ensembles) |
| `permutation_importances` | Importância por permutação nos atributos originais, medida em F1-macro |
| `shap_values` | Valores SHAP (global e local, por classe) |
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
| `test_data.py` | Tamanhos do treino/teste, mapeamento completo das classes, bloqueio do teste, ausência de `difficulty` e `attack` em `X` |
| `test_leakage.py` | Samplers não alteram a predição, todas as codificações funcionam, o scanner detecta padrões de vazamento |

Executar com `.venv/Scripts/python -m pytest`.

### `docs/_txt/`
Texto extraído de todos os PDFs (enunciado e slides). Permite localizar rapidamente trechos do material da professora para citar nas justificativas.

---

## 6. Fluxo de trabalho por etapa

```
Etapa 1 (24/10)  EDA no treino ─────────────► notebooks/01_eda.ipynb
                                               diagnóstico + plano da Etapa 2
Etapa 2 (07/11)  build_pipeline + ExperimentLog ► notebooks/02_preprocessamento.ipynb
                 (um eixo de pré-processamento por vez, CV no treino)
Etapa 3 (21/11)  ≥3 algoritmos + tuning por CV ► notebooks/03_modelagem.ipynb
                 escolha do modelo ───────────► results/final_config.json (congelado)
Etapa 4 (28/11)  load_test(final_evaluation=True) uma única vez
                 métricas + explicabilidade ──► notebooks/04_avaliacao_final.ipynb
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
3. `difficulty` nunca entra como atributo.
4. Nenhuma linha do teste é removida ou alterada.
5. Escolhas de técnica, hiperparâmetros e pesos de decisão usam apenas CV no treino.
6. O modelo final é congelado em `final_config.json` antes da avaliação no teste.
7. Testes automatizados e a auditoria verificam esses pontos continuamente.
