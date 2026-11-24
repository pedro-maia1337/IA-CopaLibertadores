# %% [markdown]
# # Etapa 2 — Experimentos de pré-processamento
#
# **Disciplina:** Inteligência Artificial — Profa. Cristiane Neri Nobre
# **Trabalho em competição:** Detecção de Intrusão em Redes (NSL-KDD, base atualizada)
# **Equipe:** _preencher_
# **Etapa 2 — entrega 07/11/2026**
#
# **Objetivo.** Implementar e comparar, de forma explícita, as técnicas de pré-processamento
# exigidas na Seção 4 do enunciado: (1) tratamento de dados redundantes/inconsistentes,
# (2) ao menos 2 técnicas de balanceamento, (3) codificação simbólico-numérica de `service`,
# (4) normalização/transformação numérica e (5) redução de dimensionalidade. Somamos dois
# problemas que a Etapa 1 encontrou na base: valores impossíveis e valores ausentes.
#
# **Resumo dos resultados** (F1-macro de CV agrupada 5-fold no treino, em **todas as linhas**, como
# a métrica oficial; regra: só troca a opção mais simples por outra se o ganho passar de 1
# desvio-padrão)
#
# - **Escala da métrica:** com os 1.702 grupos de linhas idênticas e rótulos opostos na conta, todos
#   os resultados razoáveis ficam entre 0,85 e 0,88, e nem um classificador perfeito passaria de
#   0,905; o teto é aritmético, não falha dos modelos. As
#   diferenças entre técnicas são de milésimos, e a regra de 1 dp é o que separa ganho de ruído.
# - **Redundância/inconsistência:** só o boosting ganha com a limpeza de rótulos. No HGB,
#   `drop_inconsistent` sobe de 0,8777 para 0,8821 (e o dp cai de 0,0025 para 0,0009). RF (bagging)
#   e MLP não mudam além do ruído, e reclassificar os conflitos com um modelo (`relabel_model`) não
#   supera a remoção simples.
# - **Valores impossíveis:** tratá-los como ausentes (`nan`) é ao menos tão bom quanto saturar ou
#   manter, nas duas famílias testadas.
# - **Ausentes:** NaN nativo no HGB e mediana no RF e no MLP. O indicador de ausência não ajudou,
#   apesar do padrão MNAR de `dst_host_serror_rate`.
# - **Codificação de `service`:** one-hot vence ou empata com frequência, target, ordinal e
#   one-hot com raras agrupadas. Nenhuma alternativa passa da regra.
# - **Balanceamento:** nenhuma das 7 técnicas testadas supera `none`. RUS e `class_weight` pioram.
#   Com o limiar de decisão ajustado por OOF, o desbalanceamento 16:1 já é tratado na decisão.
# - **Transformação numérica:** no MLP, `log1p` + padronização é a melhor (0,8711); sem o `log1p`
#   cai para 0,8654.
# - **Redução:** nenhuma técnica passa da regra. A seleção por árvores (`tree_select`) é a única que
#   não perde (RF: 0,8819 contra 0,8808 ± 0,0018, ganho abaixo de 1 dp). O PCA perde nas duas
#   famílias, mesmo com dados padronizados.
# - **Robustez (CV 3×5):** o ganho da limpeza de rótulos no HGB se confirma (0,8777 → 0,8815). O
#   `tree_select` no RF segue empatado com não reduzir.
#
# **Índice**
# 0. Configuração e política de dados
# 1. Protocolo experimental
# 2. Dados redundantes e inconsistentes (requisito 1)
# 3. Valores impossíveis
# 4. Valores ausentes
# 5. Codificação simbólico-numérica de `service` (requisito 3)
# 6. Balanceamento de classes (requisito 2)
# 7. Normalização e transformação numérica (requisito 4)
# 8. Redução de dimensionalidade (requisito 5)
# 9. Robustez das escolhas (CV repetida)
# 10. Tabela comparativa consolidada e configurações escolhidas
# 11. Conclusões e plano para a Etapa 3

# %% [markdown]
# ## 0. Configuração e política de dados
#
# Bibliotecas, semente global (`config.RANDOM_STATE = 42`) e o registro de experimentos.
# `ExperimentLog` grava cada configuração em `results/experiments.csv` com uma chave (hash) da
# configuração e da versão da base. Reexecutar o notebook lê do registro o que já foi calculado
# e só roda a CV do que falta. Apagar o CSV refaz todos os experimentos do zero, com o mesmo
# resultado, porque todas as sementes são fixas.

# %%
import json
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, imblearn
from IPython.display import display
from nslkdd import ExperimentLog, config, load_train
from nslkdd.data import conflict_mask
from nslkdd.models import make_model
from nslkdd.preprocessing import DomainCleaner, signed_log1p
from nslkdd.search import AXES, FAMILIES, FOLD_JOBS, coordinate_search, describe, threads

X, y, groups = load_train()
log = ExperimentLog()
np.random.seed(config.RANDOM_STATE)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")
config.FIGURES.mkdir(parents=True, exist_ok=True)

# Uma cor fixa por família de algoritmo, em todos os gráficos.
FAMILY_COLOR = {"HGB_native": "#2a78d6", "RF": "#eb6834", "MLP": "#1baf7a"}
FAMILY_NAME = {"HGB_native": "HGB (boosting)", "RF": "Random Forest", "MLP": "MLP"}

print({m.__name__: m.__version__ for m in (np, pd, sklearn, imblearn)})
print("RANDOM_STATE =", config.RANDOM_STATE, "| versão da base:", log.data_version)
print("X:", X.shape, "| classes:", y.value_counts().to_dict())

# %% [markdown]
# **Política de dados.**
# - Só `treino.csv` é usado aqui. O teste (`teste_sem_rotulo.csv`) só entra no notebook 04,
#   uma única vez, depois de congelada a configuração final. `data/raw/` (NSL-KDD original) não
#   é usada: contém as linhas do teste com os rótulos.
# - `X` tem só os 41 atributos de rede (`config.FEATURES`). As colunas de vazamento
#   (`score_ids_legado`, `tipo_alerta_soc`, `tempo_resposta_analista_min`) e as 5 cópias
#   derivadas ficaram de fora já na carga, pelos motivos mostrados na Etapa 1.
# - Toda técnica que aprende algo dos dados (remoção de redundâncias, imputação, codificação,
#   escala, redução, balanceamento) é um passo do `Pipeline` do imbalanced-learn montado por
#   `build_pipeline`. A validação cruzada recebe o pipeline inteiro, então cada passo é ajustado
#   só nos folds de treino. As contagens descritivas deste notebook (quantas linhas cada
#   estratégia removeria, quantos valores são impossíveis) não alimentam nenhum modelo.

# %% [markdown]
# ## 1. Protocolo experimental
#
# **Validação.** `StratifiedGroupKFold` com 5 folds, agrupado por duplicata: as 13.067 cópias da
# Etapa 1 nunca ficam uma no treino e outra na validação, o que inflaria a nota.
#
# **Métrica.** F1-macro em **todas as linhas** do fold de validação, cópias e conflitos incluídos,
# porque é assim que a professora calcula a métrica oficial no teste. Nos 1.702 grupos com a mesma
# linha rotulada com as duas classes (3.730 linhas), a linha recebe uma única previsão e parte das
# cópias sempre erra. Mesmo um classificador perfeito fora dos conflitos, que escolhesse o melhor
# rótulo em cada conflito, ficaria em 0,905; os modelos reais ficam perto de 0,88. Não é falha do
# modelo nem vazamento, é aritmética do ruído de rótulo. Como contraste, a coluna `f1_sem_conflito`
# mostra o mesmo modelo, com o mesmo limiar, pontuado só nas linhas únicas e sem conflito
# (`evaluation_mask`); ela não decide nada e não estima a nota.
#
# **Limiar de decisão.** O F1-macro depende do ponto de corte da probabilidade. Em cada fold, o
# limiar é escolhido nas previsões out-of-fold dos *outros* folds (todas as linhas, como a métrica),
# então o fold pontuado nunca ajuda a escolher o próprio limiar.
#
# **Busca por coordenadas.** Testar todas as combinações seria inviável (5 × 3 × 5 × 5 × 8 × 5 × 5
# opções). Variamos **um eixo por vez**, com os demais fixos na melhor escolha até então, na ordem
# do fluxo de dados: redundância → domínio → imputação → codificação → escala → balanceamento →
# redução.
#
# **Regra de decisão, declarada antes de ver os resultados.** A primeira opção de cada eixo é a
# mais simples. Outra só a substitui se superar o F1-macro dela em **mais de 1 desvio-padrão**
# entre folds. Diferença menor que isso é variação de amostragem, e a simplicidade vence.
#
# **Três famílias de algoritmos.** A melhor técnica de pré-processamento depende do modelo:
# - **HGB** (`HistGradientBoostingClassifier`): boosting, aceita NaN e categorias nativamente.
# - **Random Forest**: bagging de árvores, exige imputação e codificação numérica.
# - **MLP** (rede neural): sensível a escala e a distâncias. Sem ela, a comparação de
#   normalização seria vazia, porque árvores são invariantes a transformações monotônicas.
#
# Pontos de partida (a configuração mais simples que cada algoritmo aceita):

# %%
display(pd.DataFrame(FAMILIES).T.fillna("—"))
print("Eixos por família:", {f: [a for a, _ in AXES[f]] for f in AXES})

# %% [markdown]
# A busca completa abaixo roda (ou lê do registro) as 71 configurações. Cada CV de 5 folds leva de
# 10 a 80 segundos; tudo junto, cerca de 40 minutos na primeira execução. O mesmo código está em
# `nslkdd.search`, usado também por `tools/rodar_experimentos.py`.

# %%
best, results = coordinate_search(log, X, y, groups, verbose=False)
results.to_csv(config.RESULTS / "etapa2_resultados_por_eixo.csv", index=False)
print(f"{len(results)} configurações avaliadas em {results['segundos'].sum() / 60:.0f} min de CV.")

# %% [markdown]
# Funções de apoio: a tabela de cada eixo e um gráfico de pontos com barra de ±1 desvio-padrão,
# uma coluna por família. O ponto cheio marca a opção escolhida pela regra; a linha tracejada
# marca a opção de referência (a mais simples), e a faixa cinza, ±1 dp em torno dela, mostra a
# margem que uma alternativa precisa vencer.

# %%
COLS = ["familia", "opcao", "f1_macro", "dp", "ganho_vs_ref", "f1_sem_conflito",
        "precision_anomaly", "recall_anomaly", "limiar", "escolhida"]


def axis_table(axis):
    return results.loc[results["eixo"] == axis, COLS].reset_index(drop=True)


def plot_axis(axis, title, xlim=None):
    data = results[results["eixo"] == axis]
    families = [f for f in FAMILY_COLOR if f in set(data["familia"])]
    height = 0.42 * data.groupby("familia").size().max() + 1.3
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), height),
                             sharex=True, squeeze=False)
    for ax, family in zip(axes[0], families):
        d = data[data["familia"] == family].reset_index(drop=True)
        pos = np.arange(len(d))[::-1]
        ref, ref_sd = d.loc[0, "f1_macro"], d.loc[0, "dp"]
        ax.axvspan(ref - ref_sd, ref + ref_sd, color="#e6e5e0", zorder=0)
        ax.axvline(ref, color="#8a8980", lw=1, ls="--", zorder=1)
        color = FAMILY_COLOR[family]
        ax.errorbar(d["f1_macro"], pos, xerr=d["dp"], fmt="none", ecolor=color, elinewidth=2,
                    capsize=0, zorder=2)
        for p, (_, row) in zip(pos, d.iterrows()):
            if xlim and row["f1_macro"] < xlim[0]:  # fora da escala: seta na borda + valor
                ax.annotate(f"< {row['f1_macro']:.3f} (fora da escala)".replace(".", ","), (xlim[0], p),
                            xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=9, color="#52514c")
                continue
            ax.plot(row["f1_macro"], p, "o", ms=8, zorder=3, mec=color, mew=2,
                    mfc=color if row["escolhida"] else "white")
        ax.set_yticks(pos, d["opcao"])
        ax.set_title(FAMILY_NAME[family], fontsize=11)
        ax.set_xlabel("F1-macro (CV agrupada, ±1 dp)")
        ax.grid(axis="y", visible=False)
        if xlim:
            ax.set_xlim(*xlim)
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"pre_{axis}.png", dpi=150, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 2. Dados redundantes e inconsistentes (requisito 1)
#
# A Etapa 1 mostrou dois problemas de instâncias:
# - **Redundância:** 19% das linhas são cópias de outras (os 41 atributos idênticos). Cópias
#   inflam o peso de padrões repetidos no ajuste.
# - **Inconsistência:** o material da disciplina define instâncias inconsistentes como "duas
#   instâncias iguais com classificações diferentes". Aqui há 1.702 grupos assim, ruído de rótulo
#   injetado na base.
#
# Comparamos cinco estratégias, todas aplicadas **dentro do pipeline**, só no fold de treino:
#
# | Estratégia | O que faz |
# |---|---|
# | `none` | mantém tudo (referência) |
# | `drop_duplicates` | remove cópias com o mesmo rótulo; conflitos ficam |
# | `drop_inconsistent` | uma linha por grupo e descarta os grupos com conflito |
# | `majority_label` | uma linha por grupo com o rótulo da maioria estrita; empates saem |
# | `relabel_model` | conflitos recebem o rótulo de um HGB treinado nas linhas limpas do fold; os incertos (probabilidade < 0,8) saem |
#
# Quantas linhas cada estratégia deixaria, se aplicada ao treino inteiro (só descritivo; na CV,
# cada fold de treino aplica a sua):

# %%
first = ~pd.Series(groups).duplicated().to_numpy()
conflict = conflict_mask(groups, y)
pair_dup = pd.DataFrame({"g": groups, "y": y.to_numpy()}).duplicated().to_numpy()
votes = pd.crosstab(groups, y)
strict = votes.max(axis=1) > votes.sum(axis=1) / 2  # maioria estrita no grupo
majority_ok = pd.Series(groups).map(strict).to_numpy()
n = len(y)
redundancy_counts = pd.DataFrame({
    "linhas mantidas": [n, (~pair_dup).sum(), (first & ~conflict).sum(), (first & majority_ok).sum()],
}, index=["none", "drop_duplicates", "drop_inconsistent", "majority_label"])
redundancy_counts["% do treino"] = (100 * redundancy_counts["linhas mantidas"] / n).round(1)
display(redundancy_counts)
print(f"Grupos com conflito: {pd.Series(groups[conflict]).nunique():,} ({conflict.sum():,} linhas); "
      f"desses, {(~strict[strict.index.isin(np.unique(groups[conflict]))]).sum():,} grupos empatados.")

# %% [markdown]
# Resultado da CV nas três famílias:

# %%
display(axis_table("redundancia"))
plot_axis("redundancia", "Redundância e inconsistência de instâncias")

# %% [markdown]
# **O que se conclui.**
# - **HGB:** a limpeza de rótulos ajuda claramente. `drop_inconsistent` (0,8821 ± 0,0009),
#   `majority_label` (0,8815) e `relabel_model` (0,8814) superam `none` (0,8777 ± 0,0025) por mais
#   de 1 dp. O boosting ajusta cada árvore aos erros das anteriores, então insiste em acertar
#   exemplos com rótulo trocado e acaba memorizando o ruído, o que prejudica também as linhas
#   vizinhas, sem conflito. Removê-los antes do ajuste corrige isso. As três limpezas empatam entre
#   si; pela regra, fica a primeira que passa de 1 dp e não é superada: `drop_inconsistent`.
# - A limpeza vale mesmo pontuando todas as linhas, conflitos incluídos: os conflitos saem só do
#   **treino** de cada fold. Na validação eles continuam, e o modelo mais limpo erra menos no resto.
# - **RF:** nenhuma estratégia passa da regra (todas entre 0,8799 e 0,8816). O bagging já é robusto
#   a ruído de rótulo: cada árvore vê uma amostra diferente, e o voto dilui os exemplos trocados.
#   Fica `none`.
# - **MLP:** nenhuma limpeza supera `none` (0,8711 ± 0,0044), e `drop_duplicates` piora. Fica
#   `none`.
# - A coluna `f1_sem_conflito` (mesmo modelo e limiar, linhas únicas e sem conflito) conta a mesma
#   história numa escala maior (HGB: 0,9748 → 0,9815): o ganho da limpeza não vem das linhas em
#   conflito.
# - **`relabel_model` não compensou** em nenhuma família: recuperar parte dos conflitos com um
#   modelo não superou simplesmente removê-los. É um resultado negativo útil: a informação extra
#   não paga o risco de propagar os erros do próprio modelo.
# - `drop_duplicates` sozinho não muda nada nas árvores: o problema não são as cópias, são os
#   rótulos contraditórios.

# %% [markdown]
# ## 3. Valores impossíveis
#
# A Etapa 1 encontrou valores fora da definição dos atributos: taxas maiores que 1, contagens
# acima da janela do KDD (511 conexões, 255 hosts) e negativos em atributos não negativos. Eles
# não têm relação com a classe e parecem erro de medição. Os limites vêm da definição dos
# atributos, não de estatísticas da base, então o tratamento é por linha e não aprende nada.
#
# | Estratégia | O que faz |
# |---|---|
# | `nan` | o valor vira ausente e segue para a imputação (referência) |
# | `none` | mantém o valor como está |
# | `clip` | satura no limite (taxa 3,2 → 1,0; negativo → 0) |
#
# A referência é `nan` porque não há como recuperar o valor correto: 600 em uma taxa pode ser
# 6 × 100 ou 60 × 10. Saturar inventa um valor; tratar como ausente admite que ele se perdeu.

# %%
cleaned = DomainCleaner("nan").fit(X).transform(X)
impossible = (cleaned[config.NUMERIC].isna() & X[config.NUMERIC].notna()).sum()
impossible = impossible[impossible > 0].sort_values(ascending=False)
rows_hit = (cleaned[config.NUMERIC].isna() & X[config.NUMERIC].notna()).any(axis=1)
print(f"Linhas com algum valor impossível: {rows_hit.sum():,} ({rows_hit.mean():.1%})")
print(f"Taxa de anomaly nessas linhas: {(y[rows_hit] == 'anomaly').mean():.1%} "
      f"(base: {(y == 'anomaly').mean():.1%})")
display(impossible.head(10).rename("valores impossíveis").to_frame())

# %%
display(axis_table("dominio"))
plot_axis("dominio", "Valores impossíveis")

# %% [markdown]
# **O que se conclui.** `nan` é a melhor opção nas duas famílias testadas (HGB 0,8821; MLP 0,8711).
# No HGB, `none` e `clip` perdem menos de 0,001, dentro do ruído. No MLP, `clip` empata e manter o
# valor (`none`) perde 0,003: um valor como 600 em uma taxa vira, depois da padronização, um ponto
# extremo que distorce os pesos da rede. O RF não foi testado neste eixo: a escolha das árvores já estava coberta pelo HGB, e
# a regra manteria a referência. Fica `nan` em todas as famílias.

# %% [markdown]
# ## 4. Valores ausentes
#
# 94% das linhas têm algum ausente (até 40% em `src_bytes`). Descartar linhas é inviável, e o
# teste também tem ausentes, então o modelo precisa saber lidar com eles. Comparamos:
#
# | Estratégia | O que faz |
# |---|---|
# | `none` | NaN nativo: o HGB aprende para que lado da divisão vai o ausente (só árvores) |
# | `mean`, `median` | substitui pela média ou mediana do fold de treino |
# | `median_indicator` | mediana + uma coluna binária "estava ausente" por atributo |
# | `iterative` | estima cada atributo a partir dos outros (MICE, `IterativeImputer`) |
#
# Atributos categóricos ausentes viram a categoria `ausente` em todas as estratégias.
#
# A Etapa 1 mostrou que `dst_host_serror_rate` é MNAR (valores altos somem mais, e valor alto
# indica ataque). A mediana apaga esse sinal; o NaN nativo e o indicador o preservam.

# %%
missing = X.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]
print(f"Atributos com ausentes: {len(missing)} | linhas com algum ausente: {X.isna().any(axis=1).mean():.1%}")
col = "dst_host_serror_rate"
is_na = X[col].isna()
print(f"{col}: taxa de anomaly com ausente {(y[is_na] == 'anomaly').mean():.1%} "
      f"× sem ausente {(y[~is_na] == 'anomaly').mean():.1%}")
display((100 * missing.head(8)).round(1).rename("% ausente").to_frame())

# %%
display(axis_table("imputacao"))
plot_axis("imputacao", "Imputação de ausentes")

# %% [markdown]
# **O que se conclui.**
# - **HGB:** NaN nativo (0,8821) fica à frente da mediana (0,8817) e do indicador (0,8814).
#   Nenhuma imputação supera a referência, e o NaN nativo é o mais simples: o modelo aprende
#   sozinho o que o ausente significa em cada atributo.
# - **RF:** as quatro opções empatam (0,8807 a 0,8811). Fica a mediana, a referência.
# - **MLP:** a mediana (0,8711) fica à frente; média (0,8692) e iterativa (0,8685) perdem pouco, e
#   o indicador perde 0,012. O indicador dobra o número de colunas numéricas com sinais binários fracos, e a rede
#   ajusta ruído com eles.
# - O indicador não melhorou em nenhuma família, apesar do MNAR em `dst_host_serror_rate`. O sinal
#   do ausente nesse atributo é pequeno perto do que os outros 40 atributos já informam. Relatamos
#   o resultado assim mesmo, porque ele contraria a expectativa da Etapa 1.

# %% [markdown]
# ## 5. Codificação simbólico-numérica de `service` (requisito 3)
#
# `protocol_type` (3 valores) e `flag` (11) têm poucas categorias e usam one-hot (ou a codificação
# nativa do HGB). O desafio é `service`: 67 categorias após padronizar a grafia
# (68 contando `ausente`), com cauda longa.

# %%
svc = X["service"].fillna("ausente")
share = svc.value_counts(normalize=True)
rate = (y == "anomaly").groupby(svc).mean()
print(f"Categorias de service: {svc.nunique()} | as 5 maiores cobrem {share.head(5).sum():.1%} das linhas")
print(f"Categorias com < 0,1% das linhas (agrupadas por onehot_rare): {(share < 0.001).sum()}")
print(f"Taxa de anomaly por categoria: de {rate.min():.0%} a {rate.max():.0%}")
display(pd.DataFrame({"% linhas": (100 * share.head(8)).round(1),
                      "% anomaly": (100 * rate[share.head(8).index]).round(1)}))

# %% [markdown]
# Alternativas comparadas e o que cada uma assume:
#
# | Codificação | Colunas | Prós e contras |
# |---|---|---|
# | `onehot` | 67 | não impõe ordem (distância de Hamming constante entre categorias); muitas colunas esparsas |
# | `onehot_rare` | ~19 | one-hot com as categorias de < 0,1% das linhas em uma coluna "rara" |
# | `frequency` | 1 | pseudoatributo: fração de linhas da categoria; categorias diferentes com a mesma frequência colapsam |
# | `target` | 1 | pseudoatributo: taxa de `anomaly` da categoria, com *cross-fitting* interno para não vazar o rótulo |
# | `ordinal` | 1 | inteiro arbitrário; impõe uma ordem falsa, aceitável só para árvores |
# | nativa (HGB) | 1 | o HGB divide as categorias em grupos sem supor ordem |
#
# O material da disciplina alerta que a codificação 1-de-c não é boa opção quando há muitas
# categorias (o exemplo dos "193 países") e sugere pseudoatributos. Por isso testamos `frequency`
# e `target` contra o one-hot. O HGB usa a codificação nativa e não entra neste eixo.

# %%
display(axis_table("codificacao"))
plot_axis("codificacao", "Codificação de service")

# %% [markdown]
# **O que se conclui.**
# - **RF:** as cinco opções empatam dentro de 1 dp (0,8800 a 0,8812). Fica `onehot`, a referência.
#   O `target` é o pior: concentrar a informação de 67 categorias em um número esconde
#   combinações (`service` × `flag`) que as árvores aproveitam no one-hot.
# - **MLP:** `onehot_rare` tem a maior média (0,8730), mas o ganho (0,0019) fica abaixo de 1 dp
#   (0,0044), então fica `onehot`. `frequency` (0,8675) e `target` (0,8651) perdem. Para a rede, uma só coluna numérica força uma relação monotônica entre a categoria e
#   a saída, o que não existe: serviços com frequência parecida têm perfis de ataque opostos.
# - O alerta do material sobre alta cardinalidade se confirma só em parte. A cauda é longa (50
#   categorias com menos de 0,1% das linhas cada), mas essas colunas raras quase não pesam: agrupá-las
#   (`onehot_rare`) não muda o resultado além do ruído. Com 70 mil linhas, as 67 colunas esparsas
#   não atrapalham os modelos, e o one-hot preserva toda a informação sem ordem falsa.

# %% [markdown]
# ## 6. Balanceamento de classes (requisito 2)
#
# Apenas 5,8% das linhas são `anomaly` (16:1). O material da disciplina resume o dilema:
# *oversampling* arrisca sobreajuste e instâncias irreais; *undersampling* arrisca perder
# informação. Batista et al. (2004) recomendam SMOTE + Tomek quando há poucos positivos.
# Comparamos sete técnicas contra não balancear:
#
# | Técnica | O que faz (só no fold de treino) |
# |---|---|
# | `class_weight` | não reamostra: pesa os erros na classe rara (RF e HGB) |
# | `ros` | replica `anomaly` até 10.000 linhas |
# | `rus` | reduz `normal` a 10.000 linhas |
# | `smote` | gera `anomaly` sintéticos por interpolação entre vizinhos, até 10.000 |
# | `borderline_smote` | SMOTE só a partir dos positivos perto da fronteira |
# | `smote_tomek` | SMOTE + remoção de pares de Tomek (limpa a fronteira) |
# | `hybrid` | RUS de `normal` até 20.000 + SMOTE de `anomaly` até 5.000 |
#
# O balanceador é o último passo antes do modelo, dentro do pipeline: só o `fit` reamostra, e as
# linhas de validação nunca geram nem perdem instâncias. Por exigir dados completos, ele roda
# depois da imputação; no HGB (NaN nativo), só `class_weight` é possível.

# %%
display(axis_table("balanceamento"))
plot_axis("balanceamento", "Balanceamento de classes")

# %% [markdown]
# **O que se conclui.**
# - **Nenhuma técnica supera `none`** em nenhuma família. No RF, SMOTE + Tomek fica na frente por
#   pouco (0,8816 contra 0,8808 ± 0,0018), dentro do ruído. Escolha: `none`.
# - **RUS piora** em RF e MLP: descartar 80% dos `normal` perde informação sobre o tráfego legítimo,
#   e os falsos positivos sobem. `class_weight` também piora no RF.
# - **Por que balancear não ajuda aqui.** O F1-macro é calculado depois de escolher o limiar de
#   decisão nas previsões OOF. Reamostrar muda a probabilidade que o modelo atribui a `anomaly`,
#   mas o limiar ajustado já desloca a fronteira para o ponto que maximiza o F1-macro. A coluna
#   `limiar` mostra isso: com ROS ou SMOTE o limiar sobe, compensando a probabilidade inflada.
#   O balanceamento e o limiar resolvem o mesmo problema, e o limiar faz isso sem distorcer os
#   dados de treino.
# - **Limitação do SMOTE:** ele roda depois do one-hot e interpola as colunas binárias, gerando
#   valores fracionários (meio `http`, meio `ftp`) que não existem em conexões reais. Isso
#   explica parte do resultado fraco no MLP, sobretudo do `borderline_smote` (0,8548).

# %% [markdown]
# ## 7. Normalização e transformação numérica (requisito 4)
#
# Os volumes (`src_bytes`, `dst_bytes`, `duration`) e as contagens têm cauda muito longa. Para
# modelos baseados em distância ou gradiente, como o MLP, a escala dos atributos importa.
# Medimos a assimetria antes e depois do `log1p` (com sinal) nos atributos de cauda longa
# (`config.HEAVY_TAILED`). É uma transformação fixa, por valor, sem parâmetros aprendidos:

# %%
heavy = X[config.HEAVY_TAILED]
skew = pd.DataFrame({"assimetria original": heavy.skew(),
                     "assimetria após log1p": signed_log1p(heavy).skew()}).round(1)
display(skew.sort_values("assimetria original", ascending=False).head(8))

# %% [markdown]
# Comparamos, no MLP:
#
# | Opção | O que faz |
# |---|---|
# | `standard, log_transform=True` | `log1p` nos de cauda longa + padronização (média 0, dp 1) — referência |
# | `standard, log_transform=False` | só padronização |
# | `minmax, log_transform=True` | `log1p` + escala para [0, 1] |
# | `robust, log_transform=True` | `log1p` + mediana e intervalo interquartil |
# | `quantile, log_transform=False` | transforma cada atributo para distribuição normal por quantis |
#
# As árvores (HGB, RF) dividem por limiares e são invariantes a transformações monotônicas:
# padronizar ou aplicar `log1p` não muda nenhuma divisão possível. Por isso o eixo só é testado
# no MLP; nas árvores, a escala fica em `none`.

# %%
display(axis_table("escala"))
plot_axis("escala", "Normalização e transformação numérica (MLP)")

# %% [markdown]
# **O que se conclui.**
# - `log1p` + padronização é a melhor opção (0,8711 ± 0,0044). Sem o `log1p` cai para 0,8654: com
#   assimetria acima de 100, a padronização sozinha deixa quase todas as linhas espremidas perto
#   de zero e poucas linhas em valores enormes, e a rede não consegue separar o tráfego comum.
# - `minmax` (0,8669) sofre o mesmo efeito, porque o máximo é definido por poucas linhas extremas.
# - `robust` é o pior (0,8623): em atributos com mais da metade dos valores iguais a zero, o
#   intervalo interquartil é zero ou quase, e a escala explode.
# - `quantile` (0,8646) resolve a cauda, mas achata as diferenças entre valores grandes, que são
#   justamente as que separam alguns ataques de volume.

# %% [markdown]
# ## 8. Redução de dimensionalidade (requisito 5)
#
# Depois do one-hot, o RF e o MLP recebem cerca de 120 colunas, e a Etapa 1 mostrou grupos de
# atributos muito correlacionados (as taxas `serror`/`rerror`). Comparamos:
#
# | Técnica | O que faz |
# |---|---|
# | `none` | todas as colunas (referência) |
# | `pca` | componentes principais que explicam 95% da variância |
# | `kbest_f` | as 30 colunas com maior estatística F (ANOVA) em relação à classe |
# | `kbest_mi` | as 30 colunas com maior informação mútua com a classe |
# | `tree_select` | colunas com importância acima da mediana em um ExtraTrees ajustado no fold |
#
# Optamos por redução de dimensionalidade, e não por seleção de instâncias (o enunciado aceita
# qualquer uma das duas). A seleção de instâncias por limpeza de fronteira já aparece no eixo de
# balanceamento (`smote_tomek`), e a de redundâncias, na Seção 2.

# %%
display(axis_table("reducao"))
plot_axis("reducao", "Redução de dimensionalidade (eixo cortado em 0,85)", xlim=(0.85, 0.89))

# %% [markdown]
# **O PCA no RF (0,72) é um artefato da comparação, não da técnica.** O RF usa `scaling="none"`, e
# o PCA sem padronização é dominado pelos atributos de maior variância: `src_bytes` e `dst_bytes`,
# na casa dos milhões, ocupam os primeiros componentes, e as taxas (entre 0 e 1) praticamente
# somem. Para uma comparação justa, rodamos o PCA no RF com os dados padronizados e com `log1p`,
# como no MLP:

# %%
rf_before_reduction = {k: v for k, v in best["RF"].items() if k != "reduction"}
fair = {}
for option in [{"reduction": "none"}, {"reduction": "pca"}]:
    params = {**rf_before_reduction, "scaling": "standard", "log_transform": True, **option}
    r = log.run("etapa2_reducao_pca_justo", "RF", make_model("RF", threads("RF")), X, y, groups,
                params, n_jobs=FOLD_JOBS)
    fair[describe(option)] = {"f1_macro": r["f1_macro_mean"], "dp": r["f1_macro_std"],
                              "f1_sem_conflito": r["f1_macro_clean_mean"]}
display(pd.DataFrame(fair).T.rename_axis("RF com standard + log1p"))

# %% [markdown]
# **O que se conclui.**
# - Com os dados padronizados, o PCA no RF deixa de ser um desastre (0,8686), mas ainda perde para
#   não reduzir (0,8815, ver tabela acima). No MLP também perde (0,8655 contra 0,8711). O PCA combina os
#   atributos em direções de **maior variância**, que não são necessariamente as que separam as
#   classes, e o RF perde a capacidade de dividir por um atributo original de cada vez.
# - `kbest_f` e `kbest_mi` com 30 colunas perdem nas duas famílias: o critério univariado descarta
#   colunas que só ajudam em combinação com outras (por exemplo, categorias de `service` que
#   importam só com certas `flag`).
# - **`tree_select` é a única que não perde, mas não passa da regra:** no RF, 0,8819 ± 0,0016
#   contra 0,8808 ± 0,0018 (ganho de 0,0011, abaixo de 1 dp). No MLP, também empata (0,8724 contra
#   0,8711 ± 0,0044). A seleção por árvores avalia as colunas em conjunto e mantém só as de
#   importância acima da mediana; as descartadas (como as dummies de `service` raras) quase não são
#   usadas pelas árvores, por isso ela não perde nada.
# - Escolha: `none` nas duas famílias, pela regra. A Seção 9 confere com mais repetições se o
#   `tree_select` no RF esconde um ganho real.

# %% [markdown]
# ## 9. Robustez das escolhas (CV repetida 3×5)
#
# A busca por coordenadas faz muitas comparações, e as diferenças são de milésimos. Repetimos a CV
# 3 vezes (3 × 5 = 15 folds, com sementes diferentes) em dois casos: no HGB, a configuração de
# partida contra a escolhida (a única mudança adotada); no RF, a escolhida (igual à de partida)
# contra o `tree_select`, a alternativa que ficou mais perto da regra. O MLP não entra: a busca
# manteve a configuração de partida em todos os eixos.

# %%
robust_rows = []
pairs = {"HGB_native": [("partida", FAMILIES["HGB_native"]), ("escolhida", best["HGB_native"])],
         "RF": [("escolhida", best["RF"]),
                ("tree_select", {**best["RF"], "reduction": "tree_select"})]}
for family, configs in pairs.items():
    for label, params in configs:
        r = log.run("etapa2_robustez", family, make_model(family, threads(family)), X, y, groups,
                    params, n_repeats=3, n_jobs=FOLD_JOBS)
        robust_rows.append({"familia": family, "configuracao": label,
                            "f1_macro": r["f1_macro_mean"], "dp": r["f1_macro_std"],
                            "f1_sem_conflito": r["f1_macro_clean_mean"],
                            "segundos": r["cv_seconds"]})
robust = pd.DataFrame(robust_rows)
robust["ganho"] = (robust["f1_macro"] - robust.groupby("familia")["f1_macro"].transform("first")).round(4)
display(robust)

# %% [markdown]
# **O que se conclui.**
# - **HGB, limpeza de rótulos: confirmada.** Com 15 folds, `drop_inconsistent` fica em
#   0,8815 ± 0,0027 contra 0,8777 ± 0,0032 da partida: ganho de 0,0038, acima de 1 dp. É a mudança
#   de pré-processamento com efeito real nesta base.
# - **RF, `tree_select`: empate confirmado.** Com 15 folds, o ganho cai para 0,0003 (0,8812 contra
#   0,8809, dp 0,0030), dez vezes menor que 1 dp. O ganho de 0,0011 da busca era variação de
#   amostragem dos 5 folds. O RF fica sem redução.
# - A coluna `f1_sem_conflito` confirma a leitura nas duas famílias.

# %% [markdown]
# ## 10. Tabela comparativa consolidada e configurações escolhidas
#
# A tabela completa (71 configurações + experimentos extras) está em `results/experiments.csv` e,
# por eixo, em `results/etapa2_resultados_por_eixo.csv`. Abaixo, a melhor opção de cada eixo por
# família, com o ganho sobre a referência do eixo:

# %%
summary = results.loc[results["escolhida"], ["familia", "eixo", "opcao", "f1_macro", "dp"]]
alternatives = (results.sort_values("f1_macro", ascending=False)
                .groupby(["familia", "eixo"], sort=False).head(1)
                .set_index(["familia", "eixo"])[["opcao", "f1_macro"]]
                .rename(columns={"opcao": "maior F1 do eixo", "f1_macro": "F1 maior"}))
summary = summary.join(alternatives, on=["familia", "eixo"]).reset_index(drop=True)
display(summary)

# %% [markdown]
# Configuração final de pré-processamento de cada família. Ela segue para a Etapa 3, onde os
# algoritmos e hiperparâmetros serão comparados. O arquivo `results/melhores_configuracoes.json`
# guarda essas configurações (e, depois, os hiperparâmetros da Etapa 3).

# %%
best_file = config.RESULTS / "melhores_configuracoes.json"
stored = json.loads(best_file.read_text(encoding="utf-8")) if best_file.exists() else {}
stored.update(best)
best_file.write_text(json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8")
display(pd.DataFrame(best).T.fillna("—"))

# %%
final = results[results["escolhida"]].groupby("familia", sort=False).tail(1)[
    ["familia", "f1_macro", "dp", "f1_sem_conflito", "precision_anomaly", "recall_anomaly"]]
start = results.groupby("familia", sort=False).head(1)[["familia", "f1_macro"]]
final = final.merge(start.rename(columns={"f1_macro": "F1 da partida"}), on="familia")
final["ganho total"] = (final["f1_macro"] - final["F1 da partida"]).round(4)
display(final.set_index("familia"))

# %% [markdown]
# ## 11. Conclusões e plano para a Etapa 3
#
# | Requisito | Técnicas comparadas | Escolha | Justificativa |
# |---|---|---|---|
# | 1. Redundância/inconsistência | none, drop_duplicates, drop_inconsistent, majority_label, relabel_model | **drop_inconsistent** (HGB); none (RF, MLP) | o boosting memoriza rótulos trocados; o bagging dilui o ruído |
# | Valores impossíveis | nan, none, clip | **nan** | o valor original é irrecuperável; saturar inventa um dado |
# | Ausentes | NaN nativo, média, mediana, mediana + indicador, iterativa | **NaN nativo** (HGB); **mediana** (RF, MLP) | empates: vence a opção mais simples que o modelo aceita |
# | 2. Balanceamento | class_weight, ROS, RUS, SMOTE, Borderline-SMOTE, SMOTE+Tomek, híbrido | **nenhum** | o limiar OOF já trata o desbalanceamento; RUS perde informação |
# | 3. Codificação de `service` | one-hot, one-hot com raras, frequência, target, ordinal, nativa | **one-hot** (RF, MLP); nativa (HGB) | sem ordem falsa; 70 mil linhas sustentam 67 colunas |
# | 4. Transformação numérica | standard, minmax, robust, quantile, com e sem log1p | **log1p + standard** (MLP); nenhuma (árvores) | caudas longas; árvores são invariantes a escala |
# | 5. Redução | PCA, SelectKBest (F, MI), seleção por árvores | **nenhuma** | `tree_select` empata com não reduzir (CV 3×5); o PCA perde informação discriminante |
#
# **Lições para a Etapa 3.**
# - O ganho real do pré-processamento veio da **limpeza de rótulos** no boosting. Os demais eixos
#   pouco mudaram o resultado: nenhum outro eixo passou da regra. O teto agora depende mais do
#   algoritmo e dos hiperparâmetros do que do pré-processamento, e todos esbarram no limite de
#   0,905 imposto pelos rótulos contraditórios (os melhores modelos ficam perto de 0,88).
# - As árvores (HGB 0,8815 e RF 0,8809 na CV 3×5) estão à frente do MLP (0,8711). Na Etapa 3, comparar ao
#   menos HGB, XGBoost e RF com estas configurações, e incluir MLP, KNN e Naive Bayes como
#   contraste, com ajuste de hiperparâmetros pelo mesmo protocolo de CV agrupada.
# - Avaliar um ensemble (voto) das melhores árvores e congelar a configuração final em
#   `results/final_config.json` antes de qualquer contato com o teste.

# %% [markdown]
# ### Auditoria de vazamento deste notebook
#
# `python -m nslkdd.audit notebooks/02_preprocessamento.ipynb` deve retornar **0 erros**. Nenhuma
# célula carrega o teste, concatena bases, ajusta transformações fora do pipeline ou usa
# `fit_resample` fora dele. O `DomainCleaner` da Seção 3 é usado só para contar valores
# impossíveis: é uma regra fixa por linha, sem parâmetros aprendidos.
