# %% [markdown]
# # Etapa 3 — Construção e avaliação dos classificadores
#
# **Disciplina:** Inteligência Artificial — Profa. Cristiane Neri Nobre
# **Trabalho em competição:** Detecção de Intrusão em Redes (NSL-KDD, base atualizada)
# **Equipe:** _preencher_
# **Etapa 3 — entrega 21/11/2026**
#
# **Objetivo.** Treinar e comparar ao menos três algoritmos vistos na disciplina sobre as melhores
# combinações de pré-processamento da Etapa 2, ajustar hiperparâmetros por validação cruzada
# agrupada no treino, tentar subir o F1-macro sem vazamento e congelar a configuração final em
# `results/final_config.json`, que o notebook 04 usará para gerar a submissão.
#
# **Resumo dos resultados**
#
# Todas as medidas são F1-macro de CV agrupada por duplicata, só no treino.
#
# - **Oito algoritmos comparados** (5-fold): HGB 0,9820, RF 0,9813, XGB 0,9786, Extra Trees 0,9741,
#   MLP 0,9641, KNN 0,9518, Árvore de Decisão 0,8937, Naive Bayes 0,7763. Os ensembles de árvores
#   dominam; a árvore única e o Naive Bayes sofrem com o ruído de rótulo e com atributos correlacionados.
# - **Ajuste de hiperparâmetros** (busca aleatória com o mesmo protocolo): quase nenhum ganho na
#   primeira rodada; os padrões já estavam no platô.
# - **Tentativas de ganho:** atributos derivados por linha (+0,0007 no HGB na CV 5-fold, que não se
#   confirmou na CV 3×5), busca ampliada no HGB (melhor sorteio 0,9834), média de 5 sementes, votos
#   e dois algoritmos fora da disciplina (LightGBM e CatBoost).
# - **Finalistas em CV 3×5 (16 candidatos):** a maior média é a do voto HGB + LightGBM + CatBoost
#   (0,9832 ± 0,0024), mas dez candidatos ficam a menos de 1 dp dele. Pelo critério declarado
#   (empate vai para o mais simples), o escolhido é o **HGB da busca ampliada, com F1-macro
#   0,9830 ± 0,0023** (precisão em `anomaly` 0,980, recall 0,955): um modelo único, da disciplina,
#   empatado em média com o LightGBM e com desvio menor.
# - **Configuração congelada** em `results/final_config.json`: HGB com categorias e NaN nativos,
#   limpeza de rótulos `majority_label` e limiar de decisão ajustado por OOF.
#
# **Índice**
# 0. Configuração e política de dados
# 1. Protocolo experimental e critério de escolha
# 2. Comparação de oito algoritmos
# 3. Ajuste de hiperparâmetros
# 4. Tentativas de ganho adicional
# 5. Finalistas em CV repetida (3×5)
# 6. Tabela comparativa consolidada
# 7. Análise dos erros do modelo escolhido (OOF)
# 8. Escolha e congelamento do modelo final
# 9. Conclusões

# %% [markdown]
# ## 0. Configuração e política de dados
#
# Bibliotecas, semente global e o registro de experimentos. Como na Etapa 2, `ExperimentLog` guarda
# cada configuração em `results/experiments.csv` com uma chave (hash) do modelo, do
# pré-processamento e da versão da base. Reexecutar o notebook lê do registro o que já foi
# calculado. Os mesmos experimentos podem ser rodados fora do notebook com
# `python tools/rodar_experimentos.py etapa3` e `etapa3_extra`: script e notebook montam os
# modelos pelo mesmo módulo (`nslkdd.modeling`), então compartilham o cache.
#
# **Tempo de execução.** Calcular os 135 experimentos desta etapa do zero leva cerca de 2 horas
# em 16 núcleos (muito mais nos 2 núcleos do Colab). Com `RECALCULAR = False`, os resultados
# vêm do registro e o notebook roda em poucos minutos, mesmo com versões de bibliotecas
# diferentes das nossas: se a chave exata não bater (uma versão nova pode acrescentar
# parâmetros ao modelo), o registro é consultado por uma chave estável, que só considera os
# parâmetros diferentes do padrão. Com `RECALCULAR = True`, toda validação cruzada é refeita.

# %%
RECALCULAR = False  # True = refaz todas as validações cruzadas (~2 h em 16 núcleos)

import json
import warnings; warnings.filterwarnings("ignore")
from datetime import date
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, imblearn
import xgboost, lightgbm, catboost
from IPython.display import display
from nslkdd import ExperimentLog, build_pipeline, config, cross_val_proba, load_train
from nslkdd.data import conflict_mask, evaluation_mask
from nslkdd.decision import best_threshold, macro_f1_curve
from nslkdd.evaluation import binary_metrics, validation_report
from nslkdd.modeling import (N_SEARCH, N_SEARCH_HGB2, SEARCH, SEARCH_HGB2, SEEDS, catalog,
                             new_finalists, round2_best, run_catalog, run_finalists,
                             screen_engineering, tuned, EXTERNAL, SEARCH_EXT,
                             N_SEARCH_EXT, run_external, external_finalists)
from nslkdd.models import SoftVote, make_model
from nslkdd.preprocessing import FeatureBuilder
from nslkdd.search import FOLD_JOBS, threads

X, y, groups = load_train()
log = ExperimentLog(recompute=RECALCULAR)
np.random.seed(config.RANDOM_STATE)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")
config.FIGURES.mkdir(parents=True, exist_ok=True)

best = json.loads((config.RESULTS / "melhores_configuracoes.json").read_text(encoding="utf-8"))
cat = catalog(best)

print({m.__name__: m.__version__ for m in (np, pd, sklearn, imblearn, xgboost, lightgbm, catboost)})
print("RANDOM_STATE =", config.RANDOM_STATE, "| versão da base:", log.data_version)
print("X:", X.shape, "| classes:", y.value_counts().to_dict())

# %% [markdown]
# **Política de dados.**
# - Só `treino.csv` é usado. O teste (`teste_sem_rotulo.csv`) só entra no notebook 04, uma única
#   vez, depois do congelamento feito na Seção 8. `data/raw/` (NSL-KDD original) não é usada.
# - `X` tem só os 41 atributos de rede. As colunas de vazamento (`score_ids_legado`,
#   `tipo_alerta_soc`, `tempo_resposta_analista_min`) e as 5 cópias derivadas não entram.
# - Todo passo que aprende algo (limpeza de rótulos, imputação, codificação, escala, seleção) está
#   dentro do `Pipeline` montado por `build_pipeline`, ajustado só nos folds de treino. O limiar de
#   decisão também é escolhido só com previsões out-of-fold do treino.

# %% [markdown]
# ## 1. Protocolo experimental e critério de escolha
#
# **Validação.** O mesmo protocolo da Etapa 2: `StratifiedGroupKFold` com 5 folds, agrupado por
# duplicata (cópias nunca ficam uma no treino e outra na validação). A métrica principal é o
# F1-macro nas linhas que medem generalização (uma por grupo de cópias, sem rótulo contraditório);
# o F1-macro sobre todas as linhas (`f1_macro_noisy`) aparece como contraste.
#
# **Limiar de decisão.** Cada fold é pontuado com um limiar escolhido nas previsões out-of-fold dos
# outros folds. No modelo final, `ThresholdClassifier` faz o mesmo: aprende o limiar por OOF dentro
# do próprio treino.
#
# **Pré-processamento por algoritmo** (herdado da Etapa 2, `melhores_configuracoes.json`). Cada
# família recebe a configuração vencedora da sua busca; os algoritmos sem busca própria herdam a da
# família mais próxima:

# %%
display(pd.DataFrame(cat).T.fillna("—"))

# %% [markdown]
# **Critério de escolha do modelo final, declarado antes de ver os resultados:**
# 1. maior F1-macro médio na CV agrupada **repetida 3×5** (15 folds);
# 2. diferença menor que 1 desvio-padrão (entre folds) é empate;
# 3. no empate, vence o modelo mais simples e explicável (menos membros, menos passos).
#
# Acurácia nunca é critério: com 16 conexões normais para cada ataque (65.754 × 4.079), um
# classificador que diz sempre `normal` já tem 94% de acurácia e F1-macro de 0,48.

# %% [markdown]
# ## 2. Comparação de oito algoritmos
#
# Primeiro, cada algoritmo com hiperparâmetros padrão razoáveis (catálogo `make_model`) e o seu
# pré-processamento. Os oito cobrem as famílias vistas na disciplina:
#
# | Sigla | Algoritmo | Família |
# |---|---|---|
# | DT | Árvore de Decisão (CART) | árvore única |
# | RF | Random Forest | bagging de árvores |
# | ET | Extra Trees | bagging de árvores aleatorizadas |
# | HGB_native | Histogram Gradient Boosting (categorias e NaN nativos) | boosting |
# | XGB | XGBoost | boosting |
# | MLP | Rede neural (perceptron multicamadas) | redes neurais |
# | KNN | k vizinhos mais próximos | baseado em instâncias |
# | NB | Naive Bayes gaussiano | probabilístico |

# %%
catalog_results = run_catalog(log, X, y, groups, best, verbose=False)
board = log.leaderboard("etapa3")[["model", "f1_macro_mean", "f1_macro_std", "f1_macro_05_mean",
                                   "f1_anomaly_mean", "precision_anomaly_mean",
                                   "recall_anomaly_mean", "balanced_accuracy_mean",
                                   "threshold_mean", "cv_seconds"]]
display(board.reset_index(drop=True).round(4))

# %%
FAMILY = {"DT": "árvore", "RF": "bagging", "ET": "bagging", "HGB_native": "boosting",
          "XGB": "boosting", "MLP": "rede neural", "KNN": "instâncias", "NB": "probabilístico"}
PALETTE = {"árvore": "#8c8c8c", "bagging": "#eb6834", "boosting": "#2a78d6",
           "rede neural": "#1baf7a", "instâncias": "#9b59b6", "probabilístico": "#c9a227"}
b = board.sort_values("f1_macro_mean")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(b["model"], b["f1_macro_mean"], xerr=b["f1_macro_std"],
        color=[PALETTE[FAMILY[m]] for m in b["model"]], capsize=3)
for i, v in enumerate(b["f1_macro_mean"]):
    ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0.7, 1.0)
ax.set_xlabel("F1-macro (CV agrupada 5-fold, média ± dp)")
ax.set_title("Algoritmos com hiperparâmetros padrão")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PALETTE.values()]
ax.legend(handles, PALETTE.keys(), loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(config.FIGURES / "mod_algoritmos.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **O que se conclui.**
# - **Os ensembles de árvores dominam.** HGB (0,9820), RF (0,9813) e XGB (0,9786) ficam
#   praticamente empatados no topo, à frente de ET (0,9741). Os atributos do KDD são contagens e
#   taxas com limiares nítidos (ex.: `serror_rate` alto = varredura SYN), exatamente o que cortes
#   de árvore capturam.
# - **A árvore única (0,8937) perde 9 pontos para os ensembles.** Sem a média de bagging ou a
#   correção sequencial do boosting, ela se ajusta ao ruído de rótulo (1.702 grupos contraditórios)
#   e tem alta variância.
# - **MLP (0,9641) e KNN (0,9518)** dependem de escala e de distâncias. Mesmo com `log1p` +
#   padronização, a mistura de 67 colunas one-hot de `service` com taxas contínuas pesa contra
#   eles.
# - **Naive Bayes (0,7763 ± 0,042)** é o pior e o mais instável: supõe atributos independentes e
#   gaussianos, e os atributos de taxa são fortemente correlacionados (ex.: `serror_rate`,
#   `srv_serror_rate`, `dst_host_serror_rate`) e concentrados em 0 e 1.
# - O limiar ajustado fica bem abaixo de 0,5 nas árvores (0,2–0,4): com a classe `anomaly`
#   minoritária, baixar o corte troca alguns falsos positivos por mais acertos de ataque e aumenta
#   o F1-macro.
#
# Seguem para o ajuste de hiperparâmetros os três melhores (HGB, RF, XGB) e o MLP, que representa
# as redes neurais citadas no enunciado.

# %% [markdown]
# ## 3. Ajuste de hiperparâmetros
#
# **Busca aleatória** (`ParameterSampler`, semente 42) com o mesmo protocolo de CV agrupada e o
# mesmo registro. Não usamos `GridSearchCV` direto porque ele não aplica a máscara de avaliação
# (pontuaria também cópias e conflitos) nem o limiar cruzado. Espaços de busca:

# %%
display(pd.DataFrame({m: {k: str(v) for k, v in s.items()} for m, s in SEARCH.items()}).T.fillna("—"))
print("Sorteios por modelo:", N_SEARCH)

# %%
search_rows, hps = [], {}
for name in SEARCH:
    hps[name], trials = tuned(log, name, cat[name], X, y, groups, verbose=False)
    for hp, r in trials:
        search_rows.append({"modelo": name, "hiperparâmetros": json.dumps(hp),
                            "f1_macro": r["f1_macro_mean"], "dp": r["f1_macro_std"],
                            "segundos": r["cv_seconds"]})
search = pd.DataFrame(search_rows)
summary = search.groupby("modelo", sort=False).agg(sorteios=("f1_macro", "size"),
                                                    pior=("f1_macro", "min"),
                                                    melhor=("f1_macro", "max"))
summary["padrão (Seção 2)"] = [catalog_results[m]["f1_macro_mean"] for m in summary.index]
summary["ganho do tuning"] = (summary["melhor"] - summary["padrão (Seção 2)"]).round(4)
display(summary)
for name, hp in hps.items():
    print(f"{name:11s} melhor configuração: {hp}")

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
order = list(SEARCH)
sns.stripplot(data=search, x="f1_macro", y="modelo", order=order, size=7, alpha=0.7,
              hue="modelo", palette=[PALETTE[FAMILY[m]] for m in order], legend=False, ax=ax)
for i, m in enumerate(order):
    ax.plot(catalog_results[m]["f1_macro_mean"], i, marker="|", color="black", ms=18, mew=2)
ax.set_xlabel("F1-macro (CV 5-fold) — pontos: sorteios; traço preto: hiperparâmetros padrão")
ax.set_ylabel("")
ax.set_title("Busca aleatória de hiperparâmetros")
fig.tight_layout()
fig.savefig(config.FIGURES / "mod_busca.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **O que se conclui.** O ajuste de hiperparâmetros quase não rendeu, e é importante dizer isso:
# - **HGB:** a melhor configuração empata com o padrão (0,9820). O padrão do catálogo já estava no
#   platô.
# - **RF e XGB:** o melhor sorteio fica *abaixo* do padrão (RF 0,9795 vs 0,9813; XGB 0,9781 vs
#   0,9786). Com 6 a 12 sorteios, a busca não encontrou nada melhor que o padrão, que já vinha de
#   valores consagrados para esses algoritmos.
# - **MLP:** 0,9633 vs 0,9641, também empatado.
# - A dispersão entre sorteios (até 0,01 no HGB) é maior que a diferença entre os melhores. Os
#   hiperparâmetros importam para fugir de configurações ruins (ex.: poucas iterações com taxa de
#   aprendizado baixa), mas não diferenciam as boas.

# %% [markdown]
# ## 4. Tentativas de ganho adicional
#
# Como é uma competição, testamos mais quatro ideias para subir o F1-macro, todas sem tocar no
# teste e dentro do pipeline:
# 1. **Engenharia de atributos** (`FeatureBuilder`): razões e interações calculadas por linha a
#    partir dos 41 atributos de rede. É uma regra fixa, sem parâmetros aprendidos, então aplicá-la
#    ao teste não usa informação dele.
# 2. **Busca ampliada no HGB:** o melhor modelo é também o mais rápido (cerca de 17 s por CV), o que
#    permite uma segunda busca mais larga, incluindo amostragem de colunas (`max_features`).
# 3. **Algoritmos fora da disciplina:** LightGBM e CatBoost (Seção 4.3).
# 4. **Média de sementes e votos** entre os melhores modelos (Seção 5).
#
# ### 4.1 Engenharia de atributos
#
# Atributos criados (todos de uma mesma linha; ausentes propagam como NaN):

# %%
display(pd.DataFrame({"atributo": FeatureBuilder.NAMES, "definição": [
    "log1p(src_bytes) − log1p(dst_bytes): direção do fluxo",
    "log1p(src_bytes / (duration + 1)): taxa de envio",
    "1 se src_bytes = dst_bytes = 0 (conexão sem payload, típica de varredura)",
    "srv_count / (count + 1)",
    "serror_rate × count: nº de conexões com erro SYN na janela",
    "rerror_rate × count: nº de conexões rejeitadas na janela",
    "same_srv_rate × count", "diff_srv_rate × count",
    "dst_host_srv_count / (dst_host_count + 1)",
    "dst_host_serror_rate × dst_host_count", "dst_host_rerror_rate × dst_host_count",
    "dst_host_same_srv_rate × dst_host_count"]}))

# %% [markdown]
# A ideia: as taxas do KDD (`serror_rate` etc.) perdem a escala. Uma taxa de 100% de erro SYN em 2
# conexões e em 500 conexões são coisas muito diferentes, mas uma árvore precisa de vários cortes
# combinados em `serror_rate` e `count` para separá-las. O produto entrega isso pronto. Cada modelo
# usa os hiperparâmetros da Seção 3, com e sem os atributos novos:

# %%
eng = screen_engineering(log, X, y, groups, best, hps, verbose=False)
eng_table = pd.DataFrame([{"modelo": m, "sem atributos": eng[(m, False)]["f1_macro_mean"],
                           "dp (sem)": eng[(m, False)]["f1_macro_std"],
                           "com atributos": eng[(m, True)]["f1_macro_mean"],
                           "dp (com)": eng[(m, True)]["f1_macro_std"]}
                          for m in ("HGB_native", "XGB", "RF")]).set_index("modelo")
eng_table["ganho"] = (eng_table["com atributos"] - eng_table["sem atributos"]).round(4)
display(eng_table)

# %% [markdown]
# **O que se conclui.** Só o HGB melhora com os atributos novos (+0,0007), e menos que 1 desvio-padrão
# (0,0011). XGB e RF pioram um pouco (−0,0010 e −0,0009): colunas redundantes diluem a amostragem
# de atributos (`colsample_bytree` no XGB, `max_features` no RF), e as árvores profundas desses
# modelos já combinavam taxa e contagem sozinhas. A ideia segue como candidata no HGB e é tirada a
# limpo na CV repetida da Seção 5.
#
# ### 4.2 Busca ampliada no HGB
#
# Espaço mais largo que o da Seção 3 (mais iterações, folhas maiores, amostragem de colunas), com
# 20 sorteios sem e 20 com os atributos derivados:

# %%
display(pd.DataFrame({k: str(v) for k, v in SEARCH_HGB2.items()}, index=["valores"]).T)
round2 = round2_best(log, X, y, groups, best)
hgb2 = {e: hp for e, (hp, _) in round2.items()}
r2 = pd.DataFrame([{"atributos": e, **hp, "f1_macro": r["f1_macro_mean"], "dp": r["f1_macro_std"]}
                   for e, (_, trials) in round2.items() for hp, r in trials])
display(r2.sort_values("f1_macro", ascending=False).head(8).reset_index(drop=True))
display(r2.groupby("atributos")["f1_macro"].describe()[["count", "mean", "std", "min", "max"]])

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
sns.stripplot(data=r2, x="f1_macro", y=r2["atributos"].map({False: "sem atributos",
                                                           True: "com atributos"}),
              hue="max_features", palette="viridis", size=7, alpha=0.8, ax=ax)
ax.axvline(catalog_results["HGB_native"]["f1_macro_mean"], color="black", ls=":",
           label="HGB padrão (Seção 2)")
ax.set_xlabel("F1-macro (CV 5-fold)")
ax.set_ylabel("")
ax.set_title("Busca ampliada do HGB (40 sorteios)")
ax.legend(title="max_features", fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(config.FIGURES / "mod_busca_hgb2.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **O que se conclui.** O melhor sorteio, sem atributos derivados (`min_samples_leaf=20`,
# `max_leaf_nodes=31`, `max_iter=500`, `learning_rate=0,08`), chega a 0,9834 ± 0,0014. Ele combina
# mais regularização por folha (20 exemplos no mínimo) com uma taxa de aprendizado maior. Mas a
# mediana dos sorteios (0,9815) fica abaixo do HGB padrão: escolher o máximo de 20 sorteios dá
# vantagem a quem teve sorte no particionamento da semente 42. Com atributos, a distribuição desloca
# um pouco para cima (média de 0,9816 contra 0,9814), sem diferença real. `max_features < 1` não
# aparece no topo. Os melhores de cada grupo seguem para a confirmação em CV 3×5.
#
# ### 4.3 Algoritmos fora da disciplina: LightGBM e CatBoost
#
# Por ser uma competição, testamos também dois algoritmos que não foram vistos na disciplina, mas
# são referência em dados tabulares. O enunciado exige ao menos 3 algoritmos da disciplina
# (requisito já cumprido nas Seções 2 e 3) e não proíbe outros.
# - **LightGBM:** boosting por histogramas com crescimento das árvores *por folha* (expande sempre
#   a folha que mais reduz a perda), amostragem de linhas e colunas e categorias nativas.
# - **CatBoost:** boosting com árvores simétricas (o mesmo corte em todo o nível) e codificação
#   ordenada das categorias por estatísticas do alvo, calculada sem olhar o próprio exemplo.
#
# Os dois recebem o mesmo pré-processamento do HGB (limpeza de rótulos `majority_label`, NaN
# nativo) e tratam `protocol_type`, `service` e `flag` como categóricos (adaptador
# `NativeCategoricalBoost`). Protocolo idêntico: padrão do catálogo, depois busca aleatória
# (20 sorteios no LightGBM; 8 no CatBoost, que é cerca de 5 vezes mais lento).

# %%
display(pd.DataFrame({m: {k: str(v) for k, v in sp.items()} for m, sp in SEARCH_EXT.items()}).T)
ext_defaults, ext_searches = run_external(log, X, y, groups, best, verbose=False)
ext_hps = {n: hp for n, (hp, _) in ext_searches.items()}
ext_table = pd.DataFrame([{"modelo": n, "sorteios": len(ext_searches[n][1]),
                           "padrão": ext_defaults[n]["f1_macro_mean"],
                           "pior sorteio": min(r["f1_macro_mean"] for _, r in ext_searches[n][1]),
                           "melhor sorteio": max(r["f1_macro_mean"] for _, r in ext_searches[n][1]),
                           "segundos por CV": ext_defaults[n]["cv_seconds"]}
                          for n in EXTERNAL]).set_index("modelo")
ext_table.loc["HGB (busca 2, referência)"] = [N_SEARCH_HGB2, catalog_results["HGB_native"]["f1_macro_mean"],
                                              r2[~r2["atributos"]]["f1_macro"].min(),
                                              r2[~r2["atributos"]]["f1_macro"].max(),
                                              catalog_results["HGB_native"]["cv_seconds"]]
display(ext_table)
for n, hp in ext_hps.items():
    print(f"{n:9s} melhor configuração: {hp}")

# %% [markdown]
# **O que se conclui.**
# - **LightGBM:** o padrão (0,9818) já empata com o HGB padrão (0,9820). A busca encontra 0,9833
#   com árvores pequenas (`num_leaves=15`), muitas iterações (1.500) e metade das colunas por árvore,
#   empatado com o melhor HGB (0,9834). Faz sentido: os dois são boosting por histogramas e diferem
#   em detalhes (crescimento por folha, amostragem de linhas e colunas).
# - **CatBoost:** é o pior dos boosting (padrão 0,9709; melhor sorteio 0,9765) e o mais lento. As
#   árvores simétricas usam o mesmo corte em todo um nível, o que é uma boa regularização em bases
#   pequenas, mas limita a captura de regras muito específicas, como as dos ataques raros. A
#   codificação por estatísticas do alvo também é afetada pelo ruído de rótulo.
# - Os melhores de cada um seguem para a CV 3×5, sozinhos e em votos com o HGB.
#
# ## 5. Finalistas em CV repetida (3×5)
#
# Uma CV de 5 folds com uma semente só tem desvio-padrão de 0,001 a 0,002, da mesma ordem que as
# diferenças entre os candidatos. Escolher o máximo de 40 sorteios nessas condições tende a premiar
# a sorte do particionamento. Por isso, os finalistas são reavaliados com **3 repetições** da CV
# agrupada (sementes 42, 43 e 44; 15 folds), e é essa medida que decide (critério da Seção 1).
#
# Finalistas:
# - os quatro algoritmos da Seção 3 com os melhores hiperparâmetros (HGB, XGB, RF, MLP) e o voto
#   suave HGB + XGB + RF (média das probabilidades, cada membro com seu pré-processamento);
# - os candidatos da Seção 4: HGB com atributos, os melhores da busca ampliada (sem e com
#   atributos), a **média de 5 sementes** do melhor HGB (com `max_features = 0,7`, para que as
#   sementes gerem modelos diferentes) e o **voto** entre os dois HGB da busca ampliada;
# - os algoritmos fora da disciplina (Seção 4.3): LightGBM e CatBoost ajustados, e os votos
#   HGB + LightGBM, HGB + CatBoost e HGB + LightGBM + CatBoost.

# %%
clean = cat["HGB_native"]["redundancy"]
first_round = {n: build_pipeline(make_model(n, threads(n), **hps[n]), **cat[n]) for n in hps}
vote3 = SoftVote([(n.lower(), first_round[n]) for n in ("HGB_native", "XGB", "RF")])
candidates = [(n, p, {"redundancy": clean, "repeticoes": 3})
              for n, p in [*first_round.items(), ("Voto HGB+XGB+RF", vote3)]]
candidates += new_finalists(best, hps, hgb2)
candidates += external_finalists(best, hgb2, ext_hps)
final_results = run_finalists(log, X, y, groups, candidates, verbose=False)

final_board = pd.DataFrame([{"modelo": n, **{k: r[k] for k in [
    "f1_macro_mean", "f1_macro_std", "f1_macro_05_mean", "f1_macro_noisy_mean", "f1_anomaly_mean",
    "precision_anomaly_mean", "recall_anomaly_mean", "balanced_accuracy_mean", "threshold_mean",
    "cv_seconds"]}} for n, r in final_results.items()])
# Empate exato na média (4 casas): fica à frente o de menor desvio-padrão (mais estável).
final_board = final_board.sort_values(["f1_macro_mean", "f1_macro_std"],
                                      ascending=[False, True]).reset_index(drop=True)
leader = final_board.iloc[0]
final_board["empata com o 1º"] = (leader["f1_macro_mean"] - final_board["f1_macro_mean"]
                                  < leader["f1_macro_std"])
display(final_board.round(4))

# %%
fb = final_board.sort_values("f1_macro_mean")
fig, ax = plt.subplots(figsize=(8, 4.8))
colors = ["#2a78d6" if t else "#b0b0b0" for t in fb["empata com o 1º"]]
ax.barh(fb["modelo"], fb["f1_macro_mean"], xerr=fb["f1_macro_std"], color=colors, capsize=3)
ax.axvline(leader["f1_macro_mean"] - leader["f1_macro_std"], color="#eb6834", ls="--",
           label="1º colocado − 1 dp")
ax.set_xlim(0.955, 0.99)
ax.set_xlabel("F1-macro (CV agrupada 3×5, média ± dp)")
ax.set_title("Finalistas (azul: empatados com o 1º pelo critério de 1 dp)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(config.FIGURES / "mod_finalistas.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Aplicando o critério da Seção 1.** Entre os empatados com o 1º (diferença < 1 dp), vence o mais
# simples. A complexidade é medida, nesta ordem, por: (1) número de modelos que precisam ser
# treinados (um voto de três modelos custa três vezes mais e é mais difícil de explicar); (2) passos
# extras no pipeline (atributos derivados); (3) algoritmo fora da disciplina, que tem menos
# respaldo no material. Persistindo o empate, decide a maior média e, depois, o menor desvio.

# %%
def members(name):
    if name.startswith("Voto"):
        return name.count("+") + 1 + name.count("±")
    return len(SEEDS) if "sementes" in name else 1


tied = final_board[final_board["empata com o 1º"]].copy()
tied["modelos treinados"] = tied["modelo"].map(members)
tied["atributos derivados"] = tied["modelo"].str.contains("atributos")
tied["fora da disciplina"] = tied["modelo"].str.contains("LightGBM|CatBoost")
tied = tied.sort_values(["modelos treinados", "atributos derivados", "fora da disciplina",
                         "f1_macro_mean", "f1_macro_std"],
                        ascending=[True, True, True, False, True])
display(tied[["modelo", "f1_macro_mean", "f1_macro_std", "modelos treinados",
              "atributos derivados", "fora da disciplina", "cv_seconds"]].reset_index(drop=True))
chosen = tied.iloc[0]["modelo"]
print("Escolhido pelo critério:", chosen)

# %% [markdown]
# **O que se conclui.**
# - **Dez candidatos ficam a menos de 1 dp do 1º.** O voto HGB + LightGBM + CatBoost tem a maior média
#   (0,9832 ± 0,0024), mas o HGB da busca ampliada, o LightGBM e o voto HGB + LightGBM ficam a
#   0,0002 dele (0,9830). Essa diferença é dez vezes menor que o desvio entre folds. Pelo critério,
#   vence o mais simples: o **HGB da busca ampliada**, um modelo único da disciplina, sem passos
#   extras. Ele empata em média com o LightGBM e tem desvio menor (0,0023 contra 0,0030).
# - **O ganho do HGB da busca ampliada sobre o da primeira rodada (0,9823) está dentro do ruído.**
#   Além disso, a busca usou a partição da semente 42, que também entra na CV 3×5, o que dá um leve
#   viés otimista a favor dele.
# - **Algoritmos fora da disciplina:** o LightGBM ajustado alcança o HGB (0,9830), mas não o supera;
#   os dois são boosting por histogramas, com ideias muito parecidas. O CatBoost fica atrás
#   (0,9778), abaixo até do RF.
# - **Atributos derivados não se confirmam:** na CV 3×5, o HGB com atributos (0,9820) fica abaixo do
#   HGB sem eles (0,9823). O +0,0007 da Seção 4.1 era ruído da partição.
# - **Ensembles não ajudam de forma mensurável:** média de sementes (0,9827), votos entre HGBs
#   (0,9829), HGB + LightGBM (0,9830), HGB + CatBoost (0,9824), HGB + XGB + RF (0,9821) e o voto
#   triplo (0,9832, com 8 vezes o custo) ficam todos dentro do ruído do modelo único. Os modelos
#   fortes erram nos mesmos exemplos, então tirar a média corrige pouco. Isso indica um teto perto de
#   0,983 com os 41 atributos de rede.
# - **RF, CatBoost, XGB e MLP ficam fora do empate** (0,9802, 0,9778, 0,9780 e 0,9640).
# - O F1-macro **bruto** (todas as linhas, com cópias e conflitos) fica em 0,88 para todos: é o teto
#   imposto pelos 1.702 grupos com rótulos contraditórios, que nenhum modelo acerta.
# - **Custo do rigor:** sozinhas, sem nenhum atributo de rede, as colunas de vazamento
#   (`score_ids_legado` etc.) já alcançam F1-macro de 0,992 (Etapa 1). Abrir mão delas é a decisão correta: são posteriores ao evento e não
#   existiriam no momento de classificar uma conexão real.

# %% [markdown]
# ## 6. Tabela comparativa consolidada
#
# Linhas = estratégia de pré-processamento × algoritmo; colunas = métricas de CV agrupada. A
# primeira tabela é a comparação dos oito algoritmos (5-fold, hiperparâmetros padrão); a segunda,
# os finalistas (3×5). As duas vão para `results/` em CSV e Markdown, prontas para o relatório.

# %%
def describe_prep(p):
    """Resumo legível do pré-processamento de uma linha da tabela."""
    imput = "NaN nativo" if p.get("imputation") == "none" else p.get("imputation")
    service = "nativa" if p.get("service_encoding") == "ordinal" else p.get("service_encoding")
    parts = [f"redundância: {p.get('redundancy', 'none')}", f"ausentes: {imput}",
             f"service: {service}"]
    if p.get("scaling", "none") != "none":
        parts.append(("log1p + " if p.get("log_transform") else "") + p["scaling"])
    if p.get("reduction", "none") != "none":
        parts.append(p["reduction"] + (f" k={p['reduction_k']}" if "reduction_k" in p else ""))
    if p.get("engineered"):
        parts.append("+ atributos derivados")
    return "; ".join(parts)


METRIC_COLS = {"f1_macro_mean": "F1-macro", "f1_macro_std": "dp", "f1_anomaly_mean": "F1 anomaly",
               "f1_normal_mean": "F1 normal", "precision_anomaly_mean": "precisão anomaly",
               "recall_anomaly_mean": "recall anomaly", "balanced_accuracy_mean": "acur. balanceada",
               "accuracy_mean": "acurácia", "cv_seconds": "tempo (s)"}

table_algos = pd.DataFrame([{"algoritmo": n, "pré-processamento": describe_prep(cat[n]),
                             **{v: catalog_results[n][k] for k, v in METRIC_COLS.items()}}
                            for n in cat]).sort_values("F1-macro", ascending=False)

eng_prep = describe_prep({**cat["HGB_native"], "engineered": True})
prep_of = {n: describe_prep(cat[n]) for n in hps}
prep_of.update({"Voto HGB+XGB+RF": "cada membro com o seu", "HGB + atributos": eng_prep,
                "HGB busca 2": describe_prep(cat["HGB_native"]),
                "HGB busca 2 + atributos": eng_prep,
                "HGB busca 2, 5 sementes": describe_prep(cat["HGB_native"]),
                "Voto HGB busca 2 ± atributos": "cada membro com o seu",
                "LightGBM": describe_prep(cat["HGB_native"]),
                "CatBoost": describe_prep(cat["HGB_native"]),
                "Voto HGB + LightGBM": "cada membro com o seu",
                "Voto HGB + CatBoost": "cada membro com o seu",
                "Voto HGB + LightGBM + CatBoost": "cada membro com o seu"})
table_final = pd.DataFrame([{"modelo": n, "pré-processamento": prep_of[n],
                             **{v: r[k] for k, v in METRIC_COLS.items()}}
                            for n, r in final_results.items()]).sort_values("F1-macro",
                                                                            ascending=False)


def to_markdown(df):
    """Markdown simples (sem depender do pacote tabulate)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.0f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


for name, t in [("etapa3_algoritmos", table_algos), ("etapa3_finalistas", table_final)]:
    t.to_csv(config.RESULTS / f"tabela_{name}.csv", index=False)
    (config.RESULTS / f"tabela_{name}.md").write_text(to_markdown(t), encoding="utf-8")
display(table_algos.reset_index(drop=True).round(4))
display(table_final.reset_index(drop=True).round(4))

# %% [markdown]
# **Qual pré-processamento ajuda qual algoritmo, e por quê.**
# - **A limpeza de rótulos (`majority_label`) ajuda o boosting, mas não o bagging.** No HGB, ela
#   vale cerca de 0,007 de F1-macro (Etapa 2). O boosting ajusta cada árvore aos erros das
#   anteriores e acaba perseguindo exemplos com rótulo trocado. O RF tira a média de árvores
#   treinadas em amostras diferentes, o que já dilui o ruído.
# - **Escala e `log1p` importam para MLP e KNN, não para árvores.** Árvores são invariantes a
#   transformações monotônicas: só a ordem dos valores importa para os cortes. No MLP, o `log1p`
#   comprime as caudas de `src_bytes`/`dst_bytes`, que sem ele dominariam os gradientes.
# - **Codificação de `service`:** o HGB usa as categorias de forma nativa (agrupa serviços num
#   único corte). RF, MLP e KNN usam one-hot, que não impõe ordem falsa entre serviços.
# - **Ausentes:** só HGB e XGB tratam NaN de forma nativa (aprendem para que lado da divisão vai o
#   ausente). Os demais precisam de imputação pela mediana.
# - **Redução:** a seleção por árvores no RF e o `SelectKBest` no KNN reduzem o custo e, no KNN,
#   tiram dimensões irrelevantes do cálculo de distâncias. O PCA perdeu em todos os testes da
#   Etapa 2.
# - **Balanceamento:** nenhum. O limiar de decisão ajustado por OOF (0,24–0,42 nos finalistas) faz o
#   papel do balanceamento sem criar exemplos sintéticos.
# - **A acurácia esconde as diferenças:** ela vai de 95,4% (NB) a 99,8% (HGB), enquanto o F1-macro
#   vai de 0,78 a 0,98. A árvore única tem 98,4% de acurácia, mas só 69% de precisão em `anomaly`.

# %% [markdown]
# ## 7. Análise dos erros do modelo escolhido (OOF)
#
# Previsões out-of-fold do modelo escolhido (CV agrupada 5-fold, semente 42) mostram *onde* ele
# erra. Esta seção é só diagnóstica: não altera a escolha nem os hiperparâmetros. O limiar abaixo é
# escolhido no OOF inteiro, que é o que `ThresholdClassifier` fará no treino completo do
# notebook 04.

# %%
final_pipe = {n: p for n, p, _ in candidates}[chosen]
print("Modelo analisado:", chosen)
oof, oof_fold = cross_val_proba(final_pipe, X, y, groups, n_jobs=FOLD_JOBS)
mask = evaluation_mask(groups, y)
is_pos = y.to_numpy() == config.POSITIVE
thr = best_threshold(oof[mask], is_pos[mask])
pred = np.where(oof >= thr, config.POSITIVE, "normal")
print(f"Limiar escolhido no OOF: {thr:.2f}")
validation_report(y[mask], pred[mask], title="OOF, linhas avaliadas", save_as="mod_confusao_oof.png")
plt.show()
print(f"F1-macro em todas as linhas (com cópias e conflitos): "
      f"{binary_metrics(is_pos, pred == config.POSITIVE)['f1_macro']:.4f}")

# %% [markdown]
# Curva do F1-macro em função do limiar: mostra quanto a escolha do corte importa e se o máximo é
# estável (platô) ou um pico isolado.

# %%
grid = np.linspace(0.01, 0.99, 99)
curve = macro_f1_curve(oof[mask], is_pos[mask], grid)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(grid, curve, color="#2a78d6")
ax.axvline(thr, color="#eb6834", ls="--", label=f"limiar escolhido = {thr:.2f}")
ax.axvline(0.5, color="#8c8c8c", ls=":", label="0,5")
ax.set_ylim(max(0.9, curve.min()), 1.0)
ax.set_xlabel("limiar de P(anomaly)")
ax.set_ylabel("F1-macro (OOF)")
ax.set_title("F1-macro por limiar de decisão")
ax.legend()
fig.tight_layout()
fig.savefig(config.FIGURES / "mod_limiar.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"F1 no limiar escolhido: {curve[np.argmin(abs(grid - thr))]:.4f} | em 0,5: "
      f"{curve[np.argmin(abs(grid - 0.5))]:.4f} | faixa com F1 a até 0,002 do máximo: "
      f"{grid[curve >= curve.max() - 0.002].min():.2f}–{grid[curve >= curve.max() - 0.002].max():.2f}")

# %% [markdown]
# Onde estão os erros? Falsos negativos (ataques que passam como normais) e falsos positivos, por
# `service` e `flag`, comparados com a distribuição das linhas avaliadas:

# %%
ev = X[mask].assign(classe=y[mask].to_numpy(), prevista=pred[mask], p=oof[mask])
fn = ev[(ev.classe == config.POSITIVE) & (ev.prevista != config.POSITIVE)]
fp = ev[(ev.classe != config.POSITIVE) & (ev.prevista == config.POSITIVE)]
print(f"Falsos negativos: {len(fn)} de {is_pos[mask].sum()} ataques | "
      f"falsos positivos: {len(fp)} de {(~is_pos[mask]).sum()} normais")
for col in ["service", "flag"]:
    t = pd.DataFrame({"FN": fn[col].value_counts(), "FP": fp[col].value_counts(),
                      "% dos ataques": (100 * ev[ev.classe == config.POSITIVE][col]
                                        .value_counts(normalize=True)).round(1),
                      "% dos normais": (100 * ev[ev.classe != config.POSITIVE][col]
                                        .value_counts(normalize=True)).round(1)}).fillna(0)
    display(t.sort_values(["FN", "FP"], ascending=False).head(8))

# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.linspace(0, 1, 41)
ax.hist(ev.loc[ev.classe != config.POSITIVE, "p"], bins=bins, alpha=0.6, color="#1baf7a",
        label="normal", log=True)
ax.hist(ev.loc[ev.classe == config.POSITIVE, "p"], bins=bins, alpha=0.6, color="#eb6834",
        label="anomaly", log=True)
ax.axvline(thr, color="black", ls="--", label=f"limiar {thr:.2f}")
ax.set_xlabel("P(anomaly) out-of-fold")
ax.set_ylabel("linhas (escala log)")
ax.set_title("Separação das classes pelo modelo escolhido")
ax.legend()
fig.tight_layout()
fig.savefig(config.FIGURES / "mod_probabilidades.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **O que se conclui.**
# - **Erros:** o modelo deixa passar cerca de 4% dos ataques (74 falsos negativos em 1.817) e alarma
#   0,08% das conexões normais (42 falsos positivos em 53.247). O F1 de `normal` (0,999) é quase
#   perfeito; o F1-macro depende do F1 de `anomaly` (0,968).
# - **Onde:** 55 dos 74 falsos negativos têm `flag=SF` (conexão concluída normalmente), a flag de
#   95% do tráfego normal e de só 21% dos ataques. São ataques que imitam uma conexão comum: `private`
#   (17), `ftp_data` (11), `telnet` (6), `http` (6). O perfil (serviços de login e
#   transferência) sugere ataques de acesso remoto, cujo sinal está nos atributos de conteúdo
#   (`hot`, `num_failed_logins`), e não nas janelas de tráfego. `private` concentra também os falsos
#   positivos (19), o que mostra a sobreposição entre classes nesse serviço.
# - **Limiar:** a curva tem um platô largo: o F1-macro fica a menos de 0,002 do máximo entre 0,07 e
#   0,52. A escolha exata do limiar (0,17 no OOF inteiro, 0,27 em média nos folds) pesa pouco. O que
#   importa é sair de 0,5 na direção da classe minoritária.
# - O histograma mostra que a maioria das linhas tem probabilidade perto de 0 ou de 1. Os erros
#   vivem numa faixa estreita de exemplos ambíguos, e é por isso que mais modelos no ensemble não
#   resolvem: os erros não são aleatórios, são exemplos que os atributos não separam.
#
# ## 8. Escolha e congelamento do modelo final
#
# Aplicando o critério declarado na Seção 1 à tabela de finalistas (CV 3×5):

# %%
winner = final_board.set_index("modelo").loc[chosen]
print(f"Maior F1-macro: {leader['modelo']} = {leader['f1_macro_mean']:.4f} "
      f"± {leader['f1_macro_std']:.4f}")
print(f"Empatados com ele (diferença < 1 dp): {tied['modelo'].tolist()}")
print(f"Escolhido (mais simples entre os empatados): {chosen} = "
      f"{winner['f1_macro_mean']:.4f} ± {winner['f1_macro_std']:.4f}")

# %% [markdown]
# **Decisão: HGB da busca ampliada** (`HGB busca 2`). A maior média é a do voto
# HGB + LightGBM + CatBoost (0,9832), mas a diferença para o HGB (0,0002) é dez vezes menor que o
# desvio entre folds. Pelo critério declarado antes dos resultados, o empate vai para o mais
# simples: um modelo único, da disciplina, sem passos extras. Ele treina em cerca de 1/8 do tempo do
# voto triplo e pode ser explicado diretamente na Etapa 4. Entre os modelos únicos, ele empata em
# média com o LightGBM e tem desvio menor. Para o relatório, a leitura honesta é: *todos os
# candidatos de boosting ficam perto de 0,983 e são equivalentes dentro do ruído; escolhemos o mais
# simples.*
#
# Se a equipe preferir arriscar o voto triplo para tentar ganhar alguns décimos de milésimo no
# ranking, a troca precisa ser feita **agora**, antes do notebook 04, e justificada como desvio do
# critério declarado. Nós não recomendamos: o ganho esperado é indistinguível de zero.
#
# O arquivo abaixo descreve o modelo final por completo. O notebook 04 reconstrói o pipeline a
# partir dele, ajusta no treino inteiro (o `ThresholdClassifier` aprende o limiar por OOF dentro do
# treino) e gera a submissão. Depois deste ponto, nenhuma escolha das Etapas 2 e 3 muda.

# %%
final_name = "HGB busca 2"
assert chosen == final_name, "o escolhido mudou: rever o texto desta seção"
final_config = {
    "model": "HGB_native",
    "model_params": hgb2[False],
    "pipeline_params": {k: v for k, v in cat["HGB_native"].items() if k != "balancing"},
    "threshold": "ThresholdClassifier: limiar de maior F1-macro nas previsões OOF "
                 "(StratifiedGroupKFold 5, agrupado por duplicata) do próprio treino",
    "cv_f1_macro": float(winner["f1_macro_mean"]),
    "cv_f1_macro_std": float(winner["f1_macro_std"]),
    "cv_protocolo": "StratifiedGroupKFold 5 folds x 3 repetições (sementes 42-44), agrupado por "
                    "duplicata; avaliado em linhas únicas sem rótulo contraditório",
    "cv_threshold_medio": float(winner["threshold_mean"]),
    "criterio": "maior F1-macro em CV agrupada 3x5; empate (< 1 dp) pelo mais simples",
    "random_state": config.RANDOM_STATE,
    "features": "41 atributos de rede (config.FEATURES); sem colunas de vazamento nem derivadas",
    "data_version": log.data_version,
    "data_congelamento": date.today().isoformat(),
}
(config.RESULTS / "final_config.json").write_text(
    json.dumps(final_config, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(final_config, indent=2, ensure_ascii=False))

# %% [markdown]
# Verificação: o pipeline reconstruído a partir do arquivo é o mesmo que foi avaliado na CV (mesma
# chave no registro de experimentos).

# %%
saved = json.loads((config.RESULTS / "final_config.json").read_text(encoding="utf-8"))
rebuilt = build_pipeline(make_model(saved["model"], 1, **saved["model_params"]),
                         **saved["pipeline_params"])
found = log.lookup("etapa3_final", final_name, rebuilt, {"redundancy": clean, "repeticoes": 3},
                   n_splits=5, n_repeats=3)
print("Pipeline reconstruído = pipeline avaliado:", found is not None)
rebuilt

# %% [markdown]
# ## 9. Conclusões
#
# | Pergunta | Resposta |
# |---|---|
# | Melhor família | boosting de árvores (HGB e LightGBM empatados), seguido de RF e XGB |
# | Modelo final | HGB com categorias e NaN nativos, `majority_label`, limiar por OOF |
# | F1-macro esperado (CV 3×5) | 0,9830 ± 0,0023 (linhas únicas e sem conflito) |
# | O que mais rendeu | limpeza de rótulos no boosting (Etapa 2) e limiar de decisão |
# | O que não rendeu | balanceamento, PCA, ensembles, atributos derivados, CatBoost e boa parte do tuning |
# | Teto aparente | cerca de 0,983 com os 41 atributos; os erros restantes são ataques com `flag=SF` que imitam tráfego normal |
#
# **Expectativa para o teste.** A professora calcula o F1-macro em *todas* as linhas do teste. Se o
# teste tiver cópias e rótulos contraditórios na mesma proporção do treino, o número oficial deve
# ficar abaixo da CV limpa e mais perto do F1-macro bruto (0,88), porque nenhum modelo acerta os dois
# rótulos de uma mesma linha. Isso vale igualmente para todas as equipes e não muda a ordem dos
# modelos.
#
# **Próximo passo (Etapa 4).** O notebook 04 lê `results/final_config.json`, ajusta o
# `ThresholdClassifier(pipeline)` no treino inteiro, carrega o teste uma única vez
# (`load_test(final_evaluation=True)`), gera `results/submissao_7.csv` e registra o SHA-256 do
# arquivo. Em seguida vem a explicabilidade do modelo.
#
# ### Auditoria de vazamento deste notebook
#
# `python -m nslkdd.audit notebooks/03_modelagem.ipynb` deve retornar **0 erros**. Nenhuma célula
# carrega o teste, concatena bases ou ajusta transformações fora do pipeline. Os atributos
# derivados da Seção 4 são calculados por uma regra fixa por linha (`FeatureBuilder`, testada em
# `tests/test_leakage.py`), e o limiar de decisão é sempre escolhido em previsões out-of-fold.
