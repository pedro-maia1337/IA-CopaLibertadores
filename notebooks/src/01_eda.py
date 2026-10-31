# %% [markdown]
# # Etapa 1 — Análise exploratória e diagnóstico da base da competição
#
# **Disciplina:** Inteligência Artificial — Profa. Cristiane Neri Nobre
# **Trabalho em competição:** Detecção de Intrusão em Redes (NSL-KDD)
# **Equipe:** _preencher_
# **Etapa 1 — versão refeita para a base atualizada em 31/10/2026**
#
# **Objetivo.** Caracterizar `treino.csv` e produzir o diagnóstico que orienta a Etapa 2:
# classe-alvo, desbalanceamento, atributos categóricos, valores ausentes e impossíveis, outliers,
# redundâncias, inconsistências e fontes de vazamento de dados (Seção 7 do enunciado).
#
# **Resumo dos resultados**
#
# - 69.833 linhas e 50 colunas. O problema agora é **binário** (`normal` × `anomaly`), com 5,8% de
#   `anomaly` (16:1): "tudo normal" tem 94% de acurácia e F1-macro de apenas 0,485.
# - **Vazamento:** `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min` são cópias
#   ruidosas do rótulo, geradas depois do evento. Sem nenhum dado de rede, alcançam F1-macro 0,992;
#   os 41 atributos de rede, 0,970. As três ficam **fora do modelo**.
# - 5 colunas novas são cópias exatas de atributos existentes, com os mesmos ausentes: removidas.
# - **Ruído de rótulo:** 13.067 cópias (19% das linhas) e 1.702 grupos com a mesma linha rotulada
#   com as duas classes. Cerca de 57% desses conflitos são ataques reais.
# - 94% das linhas têm algum ausente (24 atributos, até 40%). A maioria é aleatória, mas
#   `dst_host_serror_rate` é **MNAR**: valores altos somem mais, e valores altos indicam ataque.
# - 8% das linhas têm valores impossíveis (taxas > 1, contagens multiplicadas, negativos), sem
#   relação com a classe: erro de medição, tratado como ausente.
# - `protocol_type`, `flag` e `service` têm grafias variantes (96 → 67 serviços após padronizar).
# - Validação: cópias exigem CV **agrupada**, pontuando só linhas únicas e sem conflito.
#
# **Índice**
# 0. Configuração e política de dados
# 1. Carga e visão geral
# 2. Qualidade dos dados: texto, ausentes e valores impossíveis
# 3. Classe-alvo e desbalanceamento
# 4. Atributos categóricos
# 5. Atributos numéricos e outliers
# 6. Redundância de atributos
# 7. Redundância e inconsistência de instâncias
# 8. Investigação de vazamento de dados
# 9. Protocolo de validação
# 10. Diagnóstico e plano para a Etapa 2

# %% [markdown]
# ## 0. Configuração e política de dados
#
# Bibliotecas, semente global (`config.RANDOM_STATE = 42`) e estilo dos gráficos. As cores das
# duas classes são fixas em todo o notebook.

# %%
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, imblearn
from IPython.display import display
from sklearn.metrics import roc_auc_score
from nslkdd import build_pipeline, config, cross_val_proba, load_train_frame
from nslkdd.audit import association_with_target, feature_set_gain
from nslkdd.data import conflict_mask, duplicate_groups, evaluation_mask, split_xy
from nslkdd.decision import best_threshold, macro_f1_curve
from nslkdd.evaluation import binary_metrics
from nslkdd.models import make_model
from nslkdd.preprocessing import DomainCleaner

np.random.seed(config.RANDOM_STATE)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")
config.FIGURES.mkdir(parents=True, exist_ok=True)

CLASSES = config.CLASSES
PALETTE = dict(zip(CLASSES, ["#2a78d6", "#eb6834"]))
SINGLE = PALETTE["normal"]


def savefig(fig, name):
    """Salva a figura em results/figures/eda_<name>.png (usada no relatório)."""
    fig.savefig(config.FIGURES / f"eda_{name}.png", dpi=150, bbox_inches="tight")


print({m.__name__: m.__version__ for m in (np, pd, sklearn, imblearn, sns)})
print("RANDOM_STATE =", config.RANDOM_STATE)

# %% [markdown]
# **Política de dados.** Toda a análise e toda decisão desta etapa usam **apenas `treino.csv`**.
# `teste_sem_rotulo.csv` é reservado à submissão final (notebook 04) e não é lido aqui. Os limites
# de domínio usados adiante vêm da definição dos atributos, e não de estatísticas do teste.
#
# A base original do NSL-KDD (`data/raw/`) não é usada: ela contém as linhas do teste atual com os
# rótulos, e cruzá-la com o teste seria fraude na competição.

# %% [markdown]
# ## 1. Carga e visão geral
#
# Lemos primeiro o arquivo **bruto**, para enxergar a sujeira como ela chega. O carregador
# `load_train_frame` padroniza o texto (Seção 2) e é o que usamos daí em diante.
#
# O arquivo `treino.csv.pdf` é a impressão do próprio CSV (1.205 páginas) e não acrescenta dados.

# %%
raw = pd.read_csv(config.TRAIN_FILE)
df = load_train_frame()
X, y, groups = split_xy(df)
is_anomaly = y.to_numpy() == config.POSITIVE
print("shape:", raw.shape)
print(f"memória: {raw.memory_usage(deep=True).sum() / 2**20:.1f} MiB")
raw.head()

# %% [markdown]
# Classificamos cada coluna pelo papel semântico, pois isso define o tratamento: além dos 41
# atributos do NSL-KDD, a base traz **8 colunas novas** que o enunciado não descreve. Elas não
# podem entrar no modelo sem investigação (Seções 6 e 8).

# %%
def kind_of(col):
    if col == config.TARGET:
        return "alvo"
    if col in config.LEAKY_COLUMNS or col in config.DERIVED_COLUMNS:
        return "nova (não documentada)"
    if col in config.CATEGORICAL:
        return "categórico"
    if col in config.BINARY or col == "su_attempted":
        return "binário"
    return "taxa [0,1]" if col in config.RATES else "contagem/contínuo"


types = pd.DataFrame({"dtype": raw.dtypes.astype(str), "papel": [kind_of(c) for c in raw.columns],
                      "n_valores": raw.nunique(), "ausentes_%": (raw.isna().mean() * 100).round(1)})
print(types["papel"].value_counts().to_string())
types[types["papel"] == "nova (não documentada)"]

# %% [markdown]
# ## 2. Qualidade dos dados: texto, ausentes e valores impossíveis
#
# ### 2.1 Categorias com grafias diferentes
#
# Sem padronização, `TCP` e `tcp` seriam categorias distintas na conversão simbólico-numérica.

# %%
pd.DataFrame({"bruto": raw[config.CATEGORICAL].nunique(), "padronizado": df[config.CATEGORICAL].nunique()})

# %% [markdown]
# `protocol_type` tem variação de caixa, `flag` também, e `service` tem espaços à direita. A
# padronização é uma regra fixa, que não aprende nada dos dados, e não remove linhas: por isso
# pode ser aplicada na carga, de forma idêntica ao treino e ao teste.
#
# ### 2.2 Valores ausentes
#
# O NSL-KDD original não tem ausentes; esta base tem, e vários atributos passam de 25%. O
# enunciado cita imputação como etapa a ser ajustada somente no treino.

# %%
missing = (X.isna().mean() * 100).loc[lambda s: s > 0].sort_values()
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(missing.index, missing.values, color=SINGLE, height=0.65)
for i, v in enumerate(missing.values):
    if v > 20:
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10)
ax.set(title="Valores ausentes por atributo (treino)", xlabel="% de linhas ausentes")
ax.grid(axis="y", visible=False)
savefig(fig, "ausentes")
plt.show()
print(f"{len(missing)} de 41 atributos têm ausentes; "
      f"{X.isna().any(axis=1).mean():.1%} das linhas têm ao menos um.")

# %% [markdown]
# Praticamente toda linha tem algum ausente, então **descartar linhas é inviável**. Mais
# importante: o teste também tem ausentes e não pode perder linhas, logo o modelo precisa saber
# lidar com eles em produção.
#
# ### 2.3 Mecanismo de ausência (MCAR ou MNAR)
#
# Medimos se a chance de ausência depende da classe. Se depende, o ausente carrega informação e
# um indicador de ausência pode ajudar; se não depende, é ruído.

# %%
rows = []
for col in missing.index:
    m = X[col].isna().to_numpy()
    rows.append({"atributo": col, "ausentes_%": round(m.mean() * 100, 1),
                 "anomaly_se_ausente_%": round(is_anomaly[m].mean() * 100, 1),
                 "anomaly_se_presente_%": round(is_anomaly[~m].mean() * 100, 1)})
mech = pd.DataFrame(rows)
mech["razao"] = (mech["anomaly_se_ausente_%"] / mech["anomaly_se_presente_%"]).round(2)
display(mech.sort_values("razao", ascending=False).head(5))

# A irmã srv_serror_rate está quase sempre presente e permite ver se o valor alto causa a ausência.
sister = DomainCleaner("nan").fit_transform(X)
bins = pd.cut(sister["dst_host_srv_serror_rate"], [-0.01, 0, 0.5, 0.99, 1],
              labels=["0", "(0; 0,5]", "(0,5; 0,99]", "(0,99; 1]"])
print("P(dst_host_serror_rate ausente) por faixa de dst_host_srv_serror_rate:")
print(sister.groupby(bins, observed=True)["dst_host_serror_rate"]
      .apply(lambda s: round(s.isna().mean(), 3)).to_string())

# %% [markdown]
# Na maioria dos atributos a taxa de anomalia é igual com ou sem ausente, compatível com ausência
# aleatória (MCAR). A exceção é **`dst_host_serror_rate`**: quando ausente, a taxa de anomalia
# quase dobra, e a ausência sobe de cerca de 30% para cerca de 53% quando o erro SYN é máximo. É
# um mecanismo **MNAR**: valores altos tendem a sumir, e valores altos indicam ataque.
#
# Consequência para a Etapa 2: imputar pela média ou mediana apagaria esse sinal, preenchendo
# justamente os casos altos com um valor baixo. Modelos com tratamento nativo de NaN e o indicador
# de ausência (`median_indicator`) são as alternativas a comparar.
#
# ### 2.4 Valores impossíveis pela definição do atributo
#
# Taxas devem estar em [0, 1]; `count`/`srv_count` são limitados pela janela do KDD (511) e
# `dst_host_*` satura em 255; nenhum atributo numérico pode ser negativo.

# %%
limits = pd.Series(config.DOMAIN_MAX)
above = X[limits.index].gt(limits)
below = X[config.NUMERIC].lt(0)
viol = pd.DataFrame({"acima_do_limite": above.sum(), "negativos": below.sum()}).fillna(0).astype(int)
viol = viol[viol.sum(axis=1) > 0].sort_values("acima_do_limite", ascending=False)
display(viol)
impossible = above.any(axis=1) | below.any(axis=1)
print(f"linhas com ao menos um valor impossível: {impossible.sum()} ({impossible.mean():.1%})")
print("taxa de anomalia nessas linhas:", round(is_anomaly[impossible].mean(), 3),
      "| nas demais:", round(is_anomaly[~impossible].mean(), 3))
print("valores das taxas acima de 1:", sorted(X.loc[X["rerror_rate"] > 1, "rerror_rate"].unique()))
display(X.loc[X["count"] > 511, ["count", "srv_count", "same_srv_rate"]].head(5))

# %% [markdown]
# Há valores impossíveis em poucas colunas: taxas iguais a 1,5, 2, 5 ou 10, contagens multiplicadas
# (`count = 600` com `srv_count = 6` e `same_srv_rate = 1`) e negativos em `duration`/`src_bytes`.
# A taxa de anomalia nessas linhas é a mesma das demais, logo não são sinal de ataque: são **erro
# de medição**, diferentes dos outliers legítimos da Seção 5.
#
# O fator de multiplicação não é identificável em todas as linhas (600 pode ser 6×100 ou 60×10),
# então a opção mais honesta é tratar o valor como ausente (`DomainCleaner("nan")`), e não
# "adivinhar" o valor real. A Etapa 2 compara com saturar no limite (`clip`). `su_attempted = 2`
# existe no NSL-KDD original e é mantido como contagem.

# %% [markdown]
# ## 3. Classe-alvo e desbalanceamento

# %%
counts = y.value_counts().reindex(CLASSES)
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.barh(CLASSES, counts.values, color=[PALETTE[c] for c in CLASSES], height=0.55)
for i, c in enumerate(CLASSES):
    ax.text(counts[c] + 800, i, f"{counts[c]:,}".replace(",", ".") + f"  ({counts[c] / len(y):.1%})",
            va="center", fontsize=10)
ax.set(title="Distribuição da classe-alvo (treino bruto)", xlabel="linhas", xlim=(0, 92000))
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
savefig(fig, "classes")
plt.show()

always_normal = binary_metrics(is_anomaly, np.zeros(len(y), dtype=bool))
print(f"razão de desbalanceamento: {counts['normal'] / counts['anomaly']:.1f}:1")
print(f"'tudo normal': acurácia {always_normal['accuracy']:.3f}, F1-macro {always_normal['f1_macro']:.3f}")

# %% [markdown]
# O problema agora é **binário** (`normal` × `anomaly`), e não mais de 5 classes, e o arquivo de
# submissão espera esses dois rótulos em `class_prevista`. O desbalanceamento é forte: responder
# "tudo normal" dá 94% de acurácia e F1-macro de apenas 0,485, o que confirma que acurácia não
# pode ser critério.
#
# No F1-macro binário, `normal` satura perto de 1, então **quase toda a diferença entre modelos
# vem do F1 de `anomaly`**. Isso torna o **limiar de decisão** tão importante quanto o
# balanceamento: ele será ajustado com previsões *out-of-fold* do treino.

# %% [markdown]
# ## 4. Atributos categóricos

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, config.LOW_CARDINALITY):
    tab = df.assign(c=df[col].fillna("(ausente)")).groupby("c")[config.TARGET].agg(
        taxa=lambda s: (s == config.POSITIVE).mean(), n="size").sort_values("taxa")
    labels = [f"{c} (n={n:,})".replace(",", ".") for c, n in tab["n"].items()]
    ax.barh(labels, tab["taxa"] * 100, color=PALETTE["anomaly"], height=0.6)
    ax.set(title=f"Taxa de anomaly por {col}", xlabel="% de anomaly")
    ax.grid(axis="y", visible=False)
fig.tight_layout()
savefig(fig, "protocol_flag")
plt.show()

# %% [markdown]
# `flag` é muito discriminante (`S0` e `RSTR` concentram anomalias; `RSTOS0` e `SH` têm taxas altas,
# mas poucas linhas, como mostra o `n` de cada barra), e o ausente dela tem
# taxa próxima da média, coerente com ausência aleatória. Tratamos ausente como uma categoria
# própria (`ausente`): não supõe nada sobre o valor real e funciona igual no teste.

# %%
service = df["service"].fillna("(ausente)")
top = service.value_counts().head(20)
rate = df.assign(s=service).groupby("s")[config.TARGET].apply(lambda s: (s == config.POSITIVE).mean())
freq = service.value_counts(normalize=True)
print(f"{df['service'].nunique()} serviços; {(freq < 0.001).sum()} com menos de 0,1% das linhas; "
      f"os 3 maiores somam {freq.head(3).sum():.1%}")
fig, ax = plt.subplots(figsize=(8, 6.5))
order = top.index[::-1]
ax.barh(order, rate[order] * 100, color=PALETTE["anomaly"], height=0.65)
ax.set(title="Taxa de anomaly nos 20 serviços mais frequentes", xlabel="% de anomaly")
ax.grid(axis="y", visible=False)
savefig(fig, "service_top20")
plt.show()

# %% [markdown]
# `service` tem alta cardinalidade e cauda longa: um one-hot criaria quase 70 colunas esparsas, e
# dezenas de categorias teriam pouquíssimas linhas. A Etapa 2 compara one-hot, one-hot com raras
# agrupadas, frequência, *target encoding* (com *cross-fitting*, para não vazar o rótulo) e a
# categoria nativa das árvores. `protocol_type` e `flag` têm poucos valores e ficam em one-hot,
# ou como categoria nativa nas árvores.

# %% [markdown]
# ## 5. Atributos numéricos e outliers

# %%
clean = DomainCleaner("nan").fit_transform(X)
stats = pd.DataFrame({"mediana": clean[config.NUMERIC].median(), "máximo": clean[config.NUMERIC].max(),
                      "skew": clean[config.NUMERIC].skew().round(1),
                      "%_zeros": (clean[config.NUMERIC].eq(0).mean() * 100).round(1)})
display(stats.sort_values("skew", ascending=False).head(8))
print("constantes:", [c for c in config.NUMERIC if clean[c].nunique() <= 1])

# %%
cols = ["src_bytes", "dst_bytes", "duration", "count"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, col in zip(axes, cols):
    for cls in CLASSES:
        ax.hist(np.log1p(clean.loc[y == cls, col].dropna()), bins=30, density=True, alpha=0.65,
                color=PALETTE[cls], label=cls)
    ax.set(title=f"log1p({col})", yticks=[])
    ax.grid(False)
axes[0].legend(frameon=False)
fig.tight_layout()
savefig(fig, "log_hist")
plt.show()

# %% [markdown]
# Mesmo depois de tratar os impossíveis, volumes e contagens têm cauda longa (assimetria alta,
# muitos zeros). `log1p` comprime a cauda sem eliminar instâncias; importa para MLP, KNN e SMOTE,
# que usam distâncias, e é neutro para árvores. `num_outbound_cmds` é constante e sai do modelo.
#
# **Outliers legítimos.** Antes de pensar em eliminá-los, verificamos o que eles são.

# %%
log_vals = np.log1p(clean[config.HEAVY_TAILED])
q1, q3 = log_vals.quantile(0.25), log_vals.quantile(0.75)
outlier = log_vals.gt(q3 + 3 * (q3 - q1)).any(axis=1).to_numpy()
print(f"linhas com outlier (IQR×3 em log): {outlier.mean():.1%}")
print(f"taxa de anomaly entre outliers: {is_anomaly[outlier].mean():.1%}; "
      f"entre os demais: {is_anomaly[~outlier].mean():.1%}")
print(f"fração das anomalias que uma remoção cega apagaria: {outlier[is_anomaly].mean():.1%}")

# %% [markdown]
# Ao contrário do NSL-KDD original, aqui os outliers **não** concentram anomalias (a taxa entre eles
# é até menor). Eles são simplesmente tráfego volumoso legítimo, e uma remoção cega perderia 14,6%
# das linhas e 11,8% das anomalias, classe que já é rara, sem nenhum ganho evidente. Por isso **não
# eliminamos outliers por regra estatística**: a limpeza remove só o que é impossível pela definição
# (Seção 2.4), e o `log1p` cuida da escala do resto.

# %% [markdown]
# ## 6. Redundância de atributos
#
# ### 6.1 Colunas novas derivadas
#
# Cinco colunas novas parecem cálculos sobre atributos existentes. Verificamos a igualdade exata,
# e se ao menos os ausentes diferem: se diferissem, elas poderiam recuperar valores perdidos.

# %%
derived = {"src_bytes_kb": df["src_bytes"] / 1024, "dst_bytes_kb": df["dst_bytes"] / 1024,
           "total_bytes": df["src_bytes"] + df["dst_bytes"], "same_srv_pct": df["same_srv_rate"] * 100}
rows = [{"coluna": k, "fórmula": config.DERIVED_COLUMNS[k],
         "erro_máx": float((df[k] - v).abs().max()),
         "mesmos_ausentes": bool((df[k].isna() == v.isna()).all())} for k, v in derived.items()]
display(pd.DataFrame(rows))
pd.crosstab(df["protocol_type"], df["protocolo_cod"])

# %% [markdown]
# As cinco são **cópias exatas** (erro zero), com os mesmos ausentes: `total_bytes` é ausente
# sempre que `src_bytes` é, então não permite recuperar nada, e os valores impossíveis são
# herdados. São atributos redundantes no sentido do slide "o valor pode ser deduzido de outro", e
# **ficam fora do modelo**. `config.FEATURES` não os inclui.
#
# ### 6.2 Correlação entre os 41 atributos

# %%
corr = clean[config.NUMERIC].drop(columns=config.CONSTANT_IN_TRAIN).corr(method="spearman")
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
pairs = upper[upper.abs() > 0.95].sort_values(ascending=False).round(3)
print(f"{len(pairs)} pares com |Spearman| > 0,95")
display(pairs.head(10).to_frame("spearman"))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, ax=ax,
            cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=7)
ax.set_title("Correlação de Spearman entre atributos numéricos")
savefig(fig, "correlacao")
plt.show()
print("Pearson serror_rate × srv_serror_rate:", round(clean["serror_rate"].corr(clean["srv_serror_rate"]), 3))

# %% [markdown]
# Dois pares passam de 0,95 em Spearman, e as taxas de erro seguem fortemente ligadas (`serror_rate`
# e `srv_serror_rate` têm Pearson de 0,97). Isso tem uma utilidade aqui: `srv_serror_rate` não tem
# ausentes, então, quando `serror_rate` falta, quase a mesma informação continua disponível, o que
# ajuda a explicar por que tantos ausentes não destroem o sinal. Redução de dimensionalidade é
# comparada na Etapa 2.

# %% [markdown]
# ## 7. Redundância e inconsistência de instâncias
#
# Dois conceitos do material: **redundância** (a mesma instância repetida) e **inconsistência**
# (instâncias iguais com rótulos diferentes). Comparamos pelos 41 atributos, tratando `NaN` como
# igual a `NaN`.

# %%
conflict = conflict_mask(groups, y)
sizes = pd.Series(groups).value_counts()
n_groups = len(sizes)
conflict_groups = len(np.unique(groups[conflict]))
print(f"linhas: {len(X)}  |  linhas distintas: {n_groups}  |  cópias excedentes: {len(X) - n_groups}")
print(f"tamanho dos grupos: {sizes.value_counts().sort_index().to_dict()}")
print(f"grupos com rótulos contraditórios: {conflict_groups} ({conflict.sum()} linhas)")
print(f"taxa de anomaly fora dos conflitos: {is_anomaly[~conflict].mean():.3%}; "
      f"nos conflitos: {is_anomaly[conflict].mean():.1%}")

# %% [markdown]
# **Quase 19% das linhas são cópias** (13 mil), e **1.702 grupos têm a mesma linha rotulada como
# `normal` e como `anomaly`**. A distribuição meio a meio nos conflitos é a marca de rótulos
# trocados.
#
# Isso muda o trabalho em três frentes:
# 1. **Ruído de rótulo**: treinar com conflitos ensina o modelo a errar.
# 2. **Avaliação**: nenhuma regra acerta os dois lados de um conflito, então o F1-macro sobre
#    todas as linhas tem um teto artificial e esconde as diferenças entre técnicas.
# 3. **Vazamento entre folds**: em CV comum, as cópias caem em folds diferentes (Seção 9).
#
# Quem seria o rótulo verdadeiro? Um modelo treinado só com as linhas sem conflito consegue
# decidir a maior parte dos casos, sem usar nenhuma coluna suspeita.

# %%
first = ~pd.Series(groups).duplicated().to_numpy()
keep = first & ~conflict
hgb = build_pipeline(make_model("HGB_native"), imputation="none", scaling="none", log_transform=False,
                     service_encoding="ordinal", categorical_encoding="ordinal")
model = hgb.fit(X[keep], y[keep])
doubtful = first & conflict
p_doubt = model.predict_proba(X[doubtful])[:, list(model.classes_).index(config.POSITIVE)]
confident = (p_doubt < 0.2) | (p_doubt > 0.8)
print(f"linhas distintas em conflito: {doubtful.sum()}")
print(f"decididas com confiança (p<0,2 ou p>0,8): {confident.mean():.1%}")
print(f"estimadas como ataque real: {(p_doubt >= 0.5).sum()} "
      f"de {doubtful.sum()}  |  anomalias sem conflito (linhas distintas): {is_anomaly[keep].sum()}")

# %% [markdown]
# O modelo decide 98,5% dos conflitos com confiança, e cerca de **57% deles parecem ataques reais**
# (968 linhas distintas, mais da metade das 1.817 anomalias limpas). Descartar todos os conflitos
# joga fora bastante exemplo positivo. Por isso a Etapa 2 compara, além da remoção, a estratégia
# `relabel_model`, que os recupera usando somente o fold de treino.

# %% [markdown]
# ## 8. Investigação de vazamento de dados (Seção 7 do enunciado)
#
# O enunciado pede investigar a correlação de cada atributo com a classe **antes** de treinar.
#
# ### 8.1 Associação de cada coluna com a classe

# %%
candidates = df[config.FEATURES + config.LEAKY_COLUMNS]
assoc = association_with_target(candidates, y)
display(assoc.head(10))
top = assoc.head(15).iloc[::-1]
colors = [PALETTE["anomaly"] if a in config.LEAKY_COLUMNS else SINGLE for a in top["atributo"]]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top["atributo"], top["associacao"], color=colors, height=0.65)
ax.set(title="Associação com a classe (laranja = colunas novas suspeitas)",
       xlabel="eta ou V de Cramér")
ax.grid(axis="y", visible=False)
savefig(fig, "associacao_classe")
plt.show()

# %% [markdown]
# As colunas novas `score_ids_legado` e `tipo_alerta_soc` aparecem acima de todos os atributos de
# rede. Associação alta sozinha não prova vazamento: atributos legítimos também discriminam, e por
# isso investigamos a **natureza** de cada uma.
#
# ### 8.2 Elas dizem respeito ao tráfego ou ao rótulo?

# %%
score = df["score_ids_legado"].to_numpy()
high = score > 0.5
alert = (df["tipo_alerta_soc"] == "ALERTA").to_numpy()
waited = df["tempo_resposta_analista_min"].notna().to_numpy()
print("AUC do score:", round(roc_auc_score(is_anomaly, score), 3))
print(f"tempo de resposta preenchido: {waited.mean():.1%} das linhas; anomaly quando preenchido: "
      f"{is_anomaly[waited].mean():.1%}; quando vazio: {is_anomaly[~waited].mean():.1%}")
display(pd.crosstab([high[~conflict], alert[~conflict]], y[~conflict].to_numpy(),
                    rownames=["score>0,5", "ALERTA"], colnames=["classe"]))

fig, ax = plt.subplots(figsize=(8, 3.4))
for cls in CLASSES:
    ax.hist(score[(y == cls).to_numpy()], bins=40, range=(0, 1), density=True, alpha=0.65,
            color=PALETTE[cls], label=cls)
ax.set(title="score_ids_legado por classe", xlabel="score", yticks=[])
ax.grid(False)
ax.legend(frameon=False)
savefig(fig, "score_vazamento")
plt.show()

# %% [markdown]
# `tempo_resposta_analista_min` é o caso mais claro: é o tempo que um analista levou para responder
# **depois** do evento. Essa informação não existe no momento de classificar uma conexão, e o
# simples fato de estar preenchida muda a taxa de anomalia de 1,5% para 51,7%.
#
# `tipo_alerta_soc` também é posterior: registra o resultado da triagem do SOC. A tabela mostra que
# a combinação das duas colunas separa as classes quase perfeitamente fora dos conflitos.
#
# Falta o `score_ids_legado`, que poderia ser legítimo se fosse a saída de um IDS em operação. O
# teste decisivo: um IDS calcula o score **a partir do tráfego**. Então, entre conexões normais,
# um score alto deveria corresponder a tráfego com aparência diferente. Medimos isso com um modelo
# que só vê o tráfego, com previsões *out-of-fold*.

# %%
proba_traffic, _ = cross_val_proba(hgb, X[~conflict].reset_index(drop=True),
                                   y[~conflict].reset_index(drop=True), groups[~conflict])
d = pd.DataFrame({"p_trafego": proba_traffic, "classe": y[~conflict].to_numpy(),
                  "score_alto": high[~conflict]})
display(d.groupby(["classe", "score_alto"])["p_trafego"].agg(["size", "mean", "median"]).round(3))

# %% [markdown]
# Entre as conexões `normal`, as de score alto têm a **mesma** probabilidade de ataque, vista pelo
# tráfego, das de score baixo: o score alto delas não reflete nada na conexão. Ele se comporta como
# **uma cópia ruidosa do rótulo**, com erros sorteados, e não como o resultado de analisar o tráfego.
#
# ### 8.3 Quanto cada coluna infla a validação
#
# Medimos com o mesmo modelo e a mesma CV agrupada. A linha "só colunas suspeitas" não usa nenhum
# dado de rede.

# %%
sets = {"41 atributos de rede": config.FEATURES,
        "+ score_ids_legado": config.FEATURES + ["score_ids_legado"],
        "+ tipo_alerta_soc": config.FEATURES + ["tipo_alerta_soc"],
        "+ tempo_resposta": config.FEATURES + ["tempo_resposta_analista_min"],
        "+ as três": config.FEATURES + config.LEAKY_COLUMNS,
        "só as três (sem rede)": config.LEAKY_COLUMNS}
gain = feature_set_gain(df, y, groups, sets)
gain

# %% [markdown]
# **Decisão: as três colunas são vazamento e ficam fora do modelo.** Três colunas sem nenhum dado de
# rede alcançam o desempenho do modelo completo, o que seria impossível para atributos que
# descrevem a conexão. O F1 que elas acrescentam não generaliza para o uso real, porque, em
# operação, nem o analista respondeu nem o SOC fez a triagem ainda: um detector que depende delas
# detecta o que já foi detectado.
#
# Elas também estão em `teste_sem_rotulo.csv`, então usá-las subiria o ranking. O enunciado é
# explícito: vazamento derruba a nota metodológica independentemente da posição. A remoção é
# garantida no código: `config.FEATURES` não as contém, e `tests/test_leakage.py` verifica isso.
#
# Entre os 41 atributos de rede, nenhum passa do alerta de associação de 0,9.

# %% [markdown]
# ## 9. Protocolo de validação
#
# Antes de comparar técnicas, é preciso garantir que a régua é confiável. Com 19% de cópias, a CV
# estratificada comum coloca a cópia de uma linha no treino e a original na validação: o modelo é
# "testado" no que já viu. Comparamos as duas réguas, com o mesmo modelo e sem limpeza.

# %%
mask = evaluation_mask(groups, y)


def f1_cv(cv_groups):
    proba, _ = cross_val_proba(hgb, X, y, cv_groups)
    t = best_threshold(proba[mask], is_anomaly[mask])
    return float(macro_f1_curve(proba[mask], is_anomaly[mask], np.array([t]))[0])


row_ids = np.arange(len(X))  # grupos unitários = CV estratificada comum
inflation = pd.DataFrame({"CV": ["estratificada comum", "agrupada por duplicata"],
                          "f1_macro": [round(f1_cv(row_ids), 4), round(f1_cv(groups), 4)]})
inflation

# %% [markdown]
# A CV comum dá um valor maior (+0,002). A diferença é pequena, da ordem do desvio entre folds, então
# não dá para dizer que ela distorceria uma comparação. O argumento é de correção, não de tamanho:
# as cópias tornam a validação parcialmente uma consulta ao treino, e a CV agrupada elimina isso por
# construção, sem custo. O protocolo do trabalho é:
#
# - `StratifiedGroupKFold` com o grupo de duplicatas: cópias nunca ficam em lados opostos;
# - pontuação em **uma linha por grupo, sem conflitos** (`evaluation_mask`), que é a situação do
#   teste: sem cópias e com um rótulo por linha. `f1_macro_noisy` registra o valor com tudo incluído;
# - limiar de decisão escolhido nos outros folds, nunca no que está sendo pontuado.

# %% [markdown]
# ## 10. Diagnóstico e plano para a Etapa 2
#
# | Problema | Evidência (treino) | Técnica(s) a comparar | Por quê |
# |---|---|---|---|
# | Vazamento | `score_ids_legado`, `tipo_alerta_soc`, `tempo_resposta_analista_min`: F1 alto sem nenhum dado de rede | **remover** | posteriores ao evento; cópias ruidosas do rótulo |
# | Colunas derivadas | 5 cópias exatas, mesmos ausentes | **remover** | redundantes, não recuperam nada |
# | Rótulos contraditórios | 1.702 grupos (3.730 linhas) | manter, remover duplicatas, remover inconsistentes, maioria, reclassificar por modelo | requisito 1; ruído de rótulo |
# | Cópias | 19% das linhas | CV agrupada + redundância por fold | evitam vazamento entre folds |
# | Grafias diferentes | `TCP`/`tcp`, `SF`/`sf`, `http ` | padronização na carga | mesma categoria |
# | Ausentes (94% das linhas) | 24 atributos; até 40% | NaN nativo, média, mediana, mediana + indicador, iterativa | linhas não podem ser descartadas |
# | Ausência MNAR | `dst_host_serror_rate` | NaN nativo ou indicador | a mediana apagaria o sinal |
# | Valores impossíveis | taxas > 1, contagens multiplicadas, negativos | ausente, saturar, manter | erro de medição, não ataque |
# | Desbalanceamento | 5,8% de `anomaly` (16:1) | `class_weight`, ROS, RUS, SMOTE e variantes, híbrido | requisito 2 |
# | Acurácia enganosa | "tudo normal": 94% de acurácia, F1-macro 0,485 | F1-macro + limiar OOF | métrica oficial |
# | `service` com 67 categorias | cauda longa | one-hot, raras agrupadas, frequência, target, nativa | requisito 3 |
# | Cauda longa | assimetria alta em volumes | `log1p` + Standard/MinMax/Robust/Quantile | requisito 4; afeta MLP/KNN/SMOTE |
# | Outliers legítimos | 14,6% das linhas, 11,8% das anomalias | **não** eliminar por regra estatística | tráfego real; perderia positivos raros |
# | Atributos correlacionados | grupos serror/rerror | PCA, SelectKBest, seleção por árvores | requisito 5 |
# | Atributo constante | `num_outbound_cmds` | remover | sem informação |
#
# Todos os passos que aprendem algo dos dados ficam **dentro do pipeline** e são ajustados apenas
# nos folds de treino da validação cruzada.

# %% [markdown]
# ### Auditoria de vazamento deste notebook
#
# `python -m nslkdd.audit notebooks/01_eda.ipynb` retorna **0 erros**. Os alertas são esperados: as
# citações às colunas de vazamento nas Seções 1 e 8 servem para *demonstrar* o vazamento, e elas
# não entram em nenhum `X` de modelo final.
