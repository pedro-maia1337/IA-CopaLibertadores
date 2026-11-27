# %% [markdown]
# # Etapa 4 — Submissão final, estimativa do F1-macro e explicabilidade
#
# **Disciplina:** Inteligência Artificial — Profa. Cristiane Neri Nobre
# **Trabalho em competição:** Detecção de Intrusão em Redes (NSL-KDD, base atualizada)
# **Equipe:** grupo 7 do Canvas (_integrantes: preencher_)
# **Etapa 4 — entrega 28/11/2026**
#
# **Objetivo.** Nesta etapa aplicamos o modelo que congelamos na Etapa 3
# (`results/final_config.json`) ao arquivo `teste_sem_rotulo.csv`, **uma única vez**, geramos
# `results/submissao_7.csv`, estimamos o F1-macro esperado no teste usando apenas o treino e
# explicamos como o modelo toma suas decisões.
#
# **Sobre o teste.** O teste não tem rótulos: o F1-macro oficial será calculado pela professora. Por
# isso, todas as métricas deste notebook vêm de validação cruzada no treino e são a nossa
# **estimativa** do desempenho no teste.
#
# **Resumo dos resultados**
#
# - **Modelo final:** HistGradientBoosting com tratamento nativo de categorias e de ausentes,
#   treinado após a limpeza das cópias e dos rótulos contraditórios (`drop_inconsistent`), com limiar
#   de decisão de 0,03 aprendido por validação cruzada. Usa apenas atributos de tráfego de rede (40
#   dos 41; o único excluído é constante).
# - **F1-macro estimado no teste: 0,8819 ± 0,0030** (CV agrupada 3×5, em todas as linhas, como a
#   métrica oficial). Refizemos o cálculo do zero neste notebook e obtivemos exatamente o valor da
#   Etapa 3. Por classe: `normal` precisão 0,987 / recall 0,984 / F1 0,986; `anomaly` precisão
#   0,759 / recall 0,798 / F1 0,778.
# - **Qualidade do modelo:** fora das linhas com rótulo contraditório, uma característica da própria
#   base, o modelo acerta 99,8% das conexões (F1-macro 0,983). Em todas as linhas, ele alcança 0,882
#   de um teto teórico de 0,905, ou seja, mais de 97% do máximo possível nesta base.
# - **Submissão:** gerada uma única vez, validada contra `modelo_de_submissao.csv` e reproduzível:
#   reexecutar o notebook gera o mesmo arquivo (SHA-256 na Seção 4 e em `results/submissao_7.sha256`).
# - **Explicabilidade:** o modelo se apoia em `service`, `dst_bytes`, `src_bytes`, `flag` e nas taxas
#   de erro por host, atributos clássicos da detecção de intrusão; permutação e SHAP concordam. Uma
#   árvore de apenas 4 níveis reproduz 98,4% das decisões, e confirmamos que nenhuma coluna de
#   vazamento chega ao modelo.
#
# **Índice**
# 0. Configuração e política de dados
# 1. Modelo congelado
# 2. Estimativa do F1-macro (validação cruzada no treino)
# 3. Treino do modelo final
# 4. Submissão (único uso do teste)
# 5. Explicabilidade do modelo final
# 6. Discussão dos resultados
# 7. Conclusões e checklist de entrega

# %% [markdown]
# ## 0. Configuração e política de dados
#
# Carregamos as bibliotecas, fixamos a semente global e lemos o treino: os 41 atributos de rede, o
# rótulo e o grupo de duplicatas de cada linha, que usamos para manter cópias sempre do mesmo lado
# da validação.

# %%
import json
import time
import warnings; warnings.filterwarnings("ignore")
import joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, imblearn, shap
from IPython.display import display
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from nslkdd import (ExperimentLog, ThresholdClassifier, build_pipeline, config, load_test,
                    load_train, write_submission)
from nslkdd.data import check_submission, conflict_mask, evaluation_mask
from nslkdd.decision import GRID, best_threshold, cross_val_proba, macro_f1_curve, positive_proba
from nslkdd.evaluation import binary_metrics, validation_report
from nslkdd.explain import anomaly_log_odds, shap_values, surrogate_rules, transformed
from nslkdd.models import make_model

X, y, groups = load_train()
np.random.seed(config.RANDOM_STATE)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")
config.FIGURES.mkdir(parents=True, exist_ok=True)
config.MODELS.mkdir(parents=True, exist_ok=True)
FOLD_JOBS = 5  # folds da CV em paralelo

print({m.__name__: m.__version__ for m in (np, pd, sklearn, imblearn, shap)})
print("RANDOM_STATE =", config.RANDOM_STATE)
print("X:", X.shape, "| classes:", y.value_counts().to_dict())

# %% [markdown]
# **Política de dados.**
# - Tomamos todas as decisões (pré-processamento, algoritmo, hiperparâmetros, limiar) nas Etapas 2 e
#   3, por validação cruzada no treino, e as congelamos em `results/final_config.json` antes deste
#   notebook. Aqui apenas aplicamos o que foi decidido.
# - Carregamos o teste **uma única vez**, na Seção 4, depois de treinar o modelo final, sem remover,
#   alterar ou reordenar linhas. O teste não participa de nenhuma escolha: nem do limiar, nem de
#   pseudo-rótulos.
# - Usamos como preditores apenas os 41 atributos de rede (`config.FEATURES`). Na EDA, identificamos
#   três colunas de vazamento (`score_ids_legado`, `tipo_alerta_soc`,
#   `tempo_resposta_analista_min`) e cinco cópias derivadas, e as mantivemos fora do modelo, inclusive
#   no teste, onde também aparecem. A base original (`data/raw/`) não é utilizada.

# %% [markdown]
# ## 1. Modelo congelado
#
# O arquivo `final_config.json` registra o modelo que escolhemos na Etapa 3: algoritmo,
# hiperparâmetros, pré-processamento e o F1-macro de CV que fundamentou a escolha. Reconstruímos o
# pipeline a partir dele e o conferimos com o registro de experimentos: a chave (hash dos
# parâmetros) é a mesma, o que garante que usamos exatamente o pipeline avaliado na Etapa 3.

# %%
cfg = json.loads((config.RESULTS / "final_config.json").read_text(encoding="utf-8"))


def build_final():
    """Pipeline do modelo final, montado só a partir de final_config.json."""
    return build_pipeline(make_model(cfg["model"], 1, **cfg["model_params"]),
                          **cfg["pipeline_params"])


log = ExperimentLog()
assert log.data_version == cfg["data_version"], "a base mudou desde o congelamento"
found = log.lookup("etapa3_final", "HGB busca 2", build_final(),
                   {"redundancy": cfg["pipeline_params"]["redundancy"], "repeticoes": 3},
                   n_splits=5, n_repeats=3)
print("Pipeline reconstruído = pipeline avaliado na Etapa 3:", found is not None)
display(pd.Series({**{f"modelo: {k}": v for k, v in cfg["model_params"].items()},
                   **{f"pré-proc.: {k}": v for k, v in cfg["pipeline_params"].items()},
                   "F1-macro CV 3x5 (Etapa 3)": f"{cfg['cv_f1_macro']:.4f} ± {cfg['cv_f1_macro_std']:.4f}"},
                  name="valor").to_frame())
build_final()

# %% [markdown]
# **Como o modelo funciona, passo a passo:**
# 1. `redundancy = drop_inconsistent` — **apenas no ajuste**, removemos as cópias e os grupos de
#    linhas idênticas com rótulos opostos. Por ser um *sampler* do imbalanced-learn, esse passo nunca
#    toca o teste nem o fold de validação.
# 2. `domain = nan` — valores impossíveis pela definição do atributo (taxa > 1, contagem acima da
#    janela, negativos) viram ausentes. É uma regra fixa por linha, sem nada aprendido dos dados.
# 3. Os categóricos (`protocol_type`, `service`, `flag`) recebem códigos ordinais e são tratados como
#    **categorias nativas** pelo HistGradientBoosting; os ausentes ficam como `NaN`, que o modelo
#    trata nativamente, sem imputação. `num_outbound_cmds`, constante no treino, sai.
# 4. HistGradientBoosting com os hiperparâmetros da nossa busca ampliada da Etapa 3.
# 5. Limiar de decisão aprendido por validação cruzada no treino (`ThresholdClassifier`, Seção 3).

# %% [markdown]
# ## 2. Estimativa do F1-macro (validação cruzada no treino)
#
# Como o teste não tem rótulos, estimamos o desempenho no treino, com o mesmo protocolo que usamos
# para escolher o modelo:
#
# - **`StratifiedGroupKFold` com 5 folds, repetido 3 vezes** (sementes 42, 43 e 44), estratificado
#   pela classe e **agrupado por duplicata**: linhas idênticas ficam sempre no mesmo fold. Assim, a
#   cópia de uma linha nunca é usada no treino enquanto a outra está na validação, o que garante uma
#   estimativa honesta.
# - Em cada fold, ajustamos o pipeline inteiro (limpeza, codificação e modelo) só nos outros 4 folds,
#   e o fold de fora recebe P(anomaly).
# - **Limiar cruzado:** o limiar de cada fold é o de maior F1-macro nas previsões *dos outros folds*.
#   O fold avaliado não participa da escolha do próprio limiar.
# - **F1-macro em todas as linhas do fold**, com cópias e conflitos, exatamente como a métrica
#   oficial é calculada no teste:
#   $\text{F1-macro} = \frac{1}{2}\left(F1_{\text{normal}} + F1_{\text{anomaly}}\right)$,
#   com $F1_c = \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}$ para cada classe $c$.
# - A estimativa é a média dos 15 folds, e o desvio-padrão entre folds mede a incerteza.
#
# Refazemos o cálculo do zero, sem consultar o registro de experimentos, para mostrar de onde vem o
# número.

# %%
SEEDS = [config.RANDOM_STATE + r for r in range(3)]
METRICS = ["f1_macro", "f1_normal", "f1_anomaly", "precision_normal", "recall_normal",
           "precision_anomaly", "recall_anomaly", "accuracy"]
pos = (y == config.POSITIVE).to_numpy()
conflict = conflict_mask(groups, y)

t0 = time.time()
rows, oof = [], {}
for r, seed in enumerate(SEEDS):
    proba, fold = cross_val_proba(build_final(), X, y, groups, n_splits=5, seed=seed,
                                  n_jobs=FOLD_JOBS)
    pred = np.zeros(len(X), dtype=bool)
    for k in range(5):
        own = fold == k
        thr = best_threshold(proba[~own], pos[~own])  # limiar aprendido nos outros folds
        pred[own] = proba[own] >= thr
        m = binary_metrics(pos[own], pred[own])
        rows.append({"repetição": r + 1, "semente": seed, "fold": k + 1, "linhas": int(own.sum()),
                     "limiar": thr, **{name: m[name] for name in METRICS}})
    oof[seed] = {"proba": proba, "fold": fold, "pred": pred}
folds = pd.DataFrame(rows)
print(f"CV 3x5 refeita em {time.time() - t0:.0f} s")
display(folds.round(4))

# %% [markdown]
# Cada linha acima é um fold: um modelo treinado em ~80% do treino e avaliado nos ~20% restantes,
# que ele nunca viu. Nossa estimativa final é a média desses 15 valores:

# %%
f1_mean, f1_std = folds["f1_macro"].mean(), folds["f1_macro"].std(ddof=0)
print(f"F1-macro estimado (CV agrupada 3x5, todas as linhas): {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Valor registrado no congelamento (Etapa 3):          {cfg['cv_f1_macro']:.4f} ± "
      f"{cfg['cv_f1_macro_std']:.4f}")
print(f"Faixa entre folds: {folds['f1_macro'].min():.4f} a {folds['f1_macro'].max():.4f}")
assert abs(f1_mean - cfg["cv_f1_macro"]) < 1e-3, "a CV refeita não reproduz a da Etapa 3"

per_class = pd.DataFrame({
    "precisão": [folds["precision_normal"].mean(), folds["precision_anomaly"].mean()],
    "recall": [folds["recall_normal"].mean(), folds["recall_anomaly"].mean()],
    "F1": [folds["f1_normal"].mean(), folds["f1_anomaly"].mean()],
}, index=["normal", "anomaly"])
per_class.loc["média macro"] = per_class.mean()
print("\nMétricas por classe (média dos 15 folds) — as que a professora vai divulgar:")
display(per_class.round(4))

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
sns.stripplot(data=folds, x="f1_macro", y=folds["repetição"].astype(str), hue="repetição",
              palette="deep", size=8, legend=False, ax=ax)
ax.axvline(f1_mean, color="black", ls="--", label=f"média {f1_mean:.4f}")
ax.axvspan(f1_mean - f1_std, f1_mean + f1_std, color="grey", alpha=0.15, label="± 1 dp")
ax.set_xlabel("F1-macro do fold (todas as linhas)")
ax.set_ylabel("repetição")
ax.set_title("F1-macro dos 15 folds da CV agrupada 3×5")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(config.FIGURES / "final_f1_folds.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **O que observamos.**
# - Obtivemos **0,8819 ± 0,0030**, exatamente o valor que fundamentou a escolha do modelo na Etapa 3.
#   O resultado é reprodutível e não depende do registro de experimentos.
# - Os 15 folds ficam entre 0,874 e 0,887: a variação é pequena, o que mostra um modelo estável em
#   diferentes amostras de ~14 mil linhas. O teste, com 24.333 linhas, deve variar ainda menos.
# - A classe `normal` atinge F1 de 0,986, e a classe `anomaly`, que representa apenas 5,8% das
#   conexões, chega a 0,778. Como o F1-macro dá o mesmo peso às duas classes, ele valoriza justamente
#   o desempenho na classe rara; a Seção 2.2 mostra o que limita essa classe nesta base.
# - Em `anomaly`, o recall (0,798) supera a precisão (0,759): o limiar favorece a detecção de ataques,
#   o comportamento desejável num sistema de detecção de intrusão, e cada acerto na classe rara pesa
#   mais no F1-macro.

# %% [markdown]
# ### 2.1 Matriz de confusão out-of-fold
#
# Juntando as previsões dos 5 folds da primeira repetição (semente 42), cada linha do treino recebe
# exatamente uma previsão, feita por um modelo que não a viu e com um limiar que não a viu. Com isso,
# podemos olhar o treino inteiro como se fosse um teste. O F1-macro dessa visão conjunta difere um
# pouco da média dos folds, porque os acertos e erros são somados antes de calcular a métrica.

# %%
oof42 = oof[SEEDS[0]]
oof_labels = np.where(oof42["pred"], config.POSITIVE, "normal")
fig = validation_report(y, oof_labels, title="OOF no treino (semente 42)",
                        save_as="final_confusao_oof.png")
plt.show()

# %% [markdown]
# ### 2.2 O que limita a métrica: rótulos contraditórios na base
#
# Na EDA, encontramos 1.702 grupos de linhas **idênticas nos 41 atributos, mas com rótulos opostos**
# (3.730 linhas). Como qualquer modelo dá a mesma resposta para entradas iguais, uma das linhas de
# cada grupo sempre será contada como erro, independentemente do algoritmo. Para dimensionar esse
# efeito, separamos as previsões OOF entre essas linhas e as demais e calculamos o **teto teórico**:
# o F1-macro de um classificador ideal, que acertasse tudo fora dos conflitos e escolhesse, em cada
# conflito, o rótulo da maioria (empate → `anomaly`, que rende mais F1-macro por ser a classe rara).

# %%
errors = oof42["pred"] != pos
majority = pd.Series(pos).groupby(groups).transform("mean").to_numpy() >= 0.5
ideal = np.where(conflict, majority, pos)
clean_rows = evaluation_mask(groups, y)
decomp = pd.DataFrame({
    "linhas": [int(conflict.sum()), int((~conflict).sum()), len(X)],
    "anomalias": [int(pos[conflict].sum()), int(pos[~conflict].sum()), int(pos.sum())],
    "erros do modelo": [int(errors[conflict].sum()), int(errors[~conflict].sum()),
                        int(errors.sum())],
    "F1-macro do modelo": [np.nan, binary_metrics(pos[~conflict], oof42["pred"][~conflict])["f1_macro"],
                           binary_metrics(pos, oof42["pred"])["f1_macro"]],
}, index=["em grupos contraditórios", "fora dos conflitos", "todas (métrica oficial)"])
display(decomp.round(4))
ceiling = binary_metrics(pos, ideal)
print(f"Teto teórico do F1-macro em todas as linhas: {ceiling['f1_macro']:.4f} "
      f"(F1 anomaly {ceiling['f1_anomaly']:.4f}, F1 normal {ceiling['f1_normal']:.4f}; "
      f"erros inevitáveis: {int((ideal != pos).sum())})")
print(f"Contraste — linhas únicas e sem conflito (evaluation_mask): F1-macro "
      f"{binary_metrics(pos[clean_rows], oof42['pred'][clean_rows])['f1_macro']:.4f}")

# %% [markdown]
# **O que observamos.**
# - **Fora dos grupos contraditórios, o modelo acerta 99,8% das conexões** (145 erros em 66.103
#   linhas, F1-macro 0,983). Quase todas as divergências restantes (1.706 de 1.851) estão nas 3.730
#   linhas com rótulo contraditório.
# - **O teto teórico da métrica nesta base é 0,905**: mesmo um classificador ideal erraria 1.702
#   linhas, quase todas na classe `anomaly`, que tem só 4.079 linhas, e o F1 de `anomaly` ficaria
#   limitado a 0,82. Nosso modelo atinge 0,882, **mais de 97% do máximo possível**. O valor de 0,88
#   reflete, portanto, uma característica da base, e não uma limitação do modelo.
# - O F1-macro nas linhas únicas e sem conflito (0,982) serve apenas como contraste, para medir o
#   efeito do ruído; não o usamos como estimativa da nota.

# %% [markdown]
# ### 2.3 O limiar de decisão
#
# O modelo devolve P(anomaly), e o limiar converte essa probabilidade em classe. O padrão 0,5 só é o
# melhor corte para acurácia com classes equilibradas. Aqui a métrica é F1-macro e `anomaly`
# representa 5,8% das linhas, por isso escolhemos o limiar por validação cruzada. A curva abaixo
# mostra o F1-macro OOF (semente 42) em cada limiar, em todas as linhas e só fora dos conflitos.

# %%
p42 = oof42["proba"]
grid = np.unique(np.r_[np.geomspace(0.001, 0.01, 7), GRID])
curve_all = macro_f1_curve(p42, pos, grid)
curve_clean = macro_f1_curve(p42[~conflict], pos[~conflict], grid)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].plot(grid, curve_all, label="todas as linhas (métrica oficial)")
axes[0].plot(grid, curve_clean - (curve_clean.max() - curve_all.max()), ls="--",
             label="fora dos conflitos (deslocada para comparar a forma)")
axes[0].axvline(best_threshold(p42, pos), color="black", ls=":", label="limiar escolhido")
axes[0].axvline(0.5, color="grey", ls=":", label="0,5")
axes[0].set_xscale("log")
axes[0].set_xlabel("limiar (escala log)")
axes[0].set_ylabel("F1-macro OOF")
axes[0].set_title("F1-macro × limiar: um platô largo")
axes[0].legend(fontsize=8)
bins = np.linspace(0, 1, 51)
for cls, color in [("normal", "#1baf7a"), (config.POSITIVE, "#eb6834")]:
    axes[1].hist(p42[(y == cls).to_numpy()], bins=bins, alpha=0.6, color=color, label=cls, log=True)
axes[1].set_xlabel("P(anomaly) out-of-fold")
axes[1].set_ylabel("linhas (escala log)")
axes[1].set_title("O modelo é muito confiante: quase tudo perto de 0 ou de 1")
axes[1].legend()
fig.tight_layout()
fig.savefig(config.FIGURES / "final_limiar.png", dpi=150, bbox_inches="tight")
plt.show()

table = []
for t in [0.01, 0.03, 0.05, 0.08, 0.2, 0.5]:
    a = binary_metrics(pos, p42 >= t)
    c = binary_metrics(pos[~conflict], p42[~conflict] >= t)
    table.append({"limiar": t, "F1-macro (todas)": a["f1_macro"],
                  "F1-macro (fora dos conflitos)": c["f1_macro"],
                  "precisão anomaly (fora)": c["precision_anomaly"],
                  "recall anomaly (fora)": c["recall_anomaly"],
                  "previstas anomaly": int((p42 >= t).sum())})
display(pd.DataFrame(table).round(4))
band = (p42 >= 0.03) & (p42 < 0.5)
print("Linhas com P(anomaly) entre 0,03 e 0,5:",
      {"normal fora dos conflitos": int((band & ~pos & ~conflict).sum()),
       "anomaly fora dos conflitos": int((band & pos & ~conflict).sum()),
       "em conflitos": int((band & conflict).sum())})

# %% [markdown]
# **O que observamos.**
# - **O limiar ficou em 0,03 porque o modelo é muito confiante.** O histograma mostra quase todas as
#   linhas com P(anomaly) perto de 0 ou de 1. Entre 0,03 e 0,5 há só 131 linhas (40 normais e 46
#   ataques fora dos conflitos, e 45 em conflitos). Ajustar o corte afeta apenas essas linhas, e nelas
#   há mais ataques do que conexões normais.
# - Como o modelo aprende com 16 conexões normais para cada ataque, ele tende a ser conservador nos
#   casos duvidosos. O limiar ajustado compensa essa tendência diretamente na decisão, sem reamostrar
#   os dados. Isso explica por que, na Etapa 2, as técnicas de balanceamento (SMOTE, undersampling,
#   class_weight) não trouxeram ganho adicional.
# - **O efeito não depende dos conflitos:** mesmo fora deles o melhor limiar é baixo (0,08), com o
#   mesmo ganho (de 0,9817 em 0,5 para 0,983 entre 0,03 e 0,08).
# - **A curva forma um platô:** entre 0,01 e 0,08 o F1-macro varia menos de 0,0005. O resultado é
#   robusto à escolha exata do limiar, o que dá segurança caso o teste seja um pouco diferente do
#   treino.

# %% [markdown]
# ### 2.4 O que esperamos no teste
#
# Nossa estimativa supõe que o teste venha da mesma distribuição do treino, inclusive com a mesma
# proporção de rótulos contraditórios. Sem os rótulos não é possível verificar isso, então
# consideramos três cenários:
#
# | Cenário do teste | F1-macro esperado |
# |---|---|
# | Gerado como o treino (cópias e rótulos contraditórios na mesma proporção) | **≈ 0,88** (faixa dos folds: 0,874 a 0,887) |
# | Com menos conflitos ou rótulos limpos | acima de 0,88, até perto de 0,98 (F1 fora dos conflitos) |
# | Com distribuição diferente (outros ataques, outro perfil de tráfego) | abaixo; fora do alcance de qualquer validação no treino |
#
# Adotamos o primeiro cenário como hipótese de trabalho. Por rigor, registramos que o valor de 0,8819
# corresponde à melhor de 20 configurações avaliadas nos mesmos folds, o que pode torná-lo
# ligeiramente otimista (da ordem de 0,001, bem menor que o desvio entre folds). Como o ruído de
# rótulo é o mesmo para todas as equipes, ele afeta o valor absoluto, mas não a comparação entre
# modelos.

# %% [markdown]
# ## 3. Treino do modelo final
#
# Ajustamos o modelo final no **treino completo** (69.833 linhas). O `ThresholdClassifier`:
# 1. faz uma CV agrupada de 5 folds no treino e escolhe o limiar de maior F1-macro nas previsões
#    out-of-fold, em todas as linhas;
# 2. ajusta o pipeline no treino inteiro;
# 3. na previsão, classifica como `anomaly` quando P(anomaly) ≥ limiar.
#
# Salvamos o modelo ajustado em `results/models/final.joblib`.

# %%
t0 = time.time()
final = ThresholdClassifier(build_final(), n_splits=5, n_jobs=FOLD_JOBS).fit(X, y)
print(f"Ajustado em {time.time() - t0:.0f} s | limiar aprendido: {final.threshold_:.2f}")
assert np.isclose(final.threshold_, best_threshold(p42, pos)), "limiar diferente do OOF da semente 42"
joblib.dump(final, config.MODELS / "final.joblib")
inner = final.estimator_
print("Linhas usadas no ajuste após drop_inconsistent:",
      inner.named_steps["redundancy"].fit_resample(X, y)[0].shape[0], "de", len(X))

# %% [markdown]
# O limiar aprendido no treino completo coincide com o da Seção 2.3 (a CV interna usa a mesma
# semente), o que confirma a consistência do processo. A remoção de cópias e conflitos acontece só no
# ajuste: o teste passa inteiro pelo modelo.
#
# ## 4. Submissão (único uso do teste)
#
# Carregamos o teste agora, **pela primeira e única vez**: `load_test(final_evaluation=True)` devolve
# os 41 atributos de rede e os ids, sem remover nem reordenar linhas, e descarta na leitura as
# colunas de vazamento que existem no arquivo. O modelo já ajustado apenas faz `predict`: nada é
# aprendido do teste.

# %%
# [TESTE]
X_te, ids = load_test(final_evaluation=True)
assert list(X_te.columns) == config.FEATURES
print("Teste:", X_te.shape, "| ids únicos:", ids.nunique())
y_te = final.predict(X_te)
submission = write_submission(ids, y_te)
sha256 = check_submission()
(config.RESULTS / f"submissao_{config.TEAM_NUMBER}.sha256").write_text(
    f"{sha256}  {config.SUBMISSION_FILE.name}\n", encoding="utf-8")
print(f"Arquivo: results/{config.SUBMISSION_FILE.name}")
print(f"SHA-256: {sha256}")
display(submission.head())

# %% [markdown]
# Antes de gravar, `write_submission` confere o formato exigido na Seção 3.1 do enunciado contra
# `modelo_de_submissao.csv`: colunas `id,class_prevista`, mesmos ids na mesma ordem, sem repetidos
# nem vazios, rótulos `normal`/`anomaly` em minúsculas. Em seguida, `check_submission` relê o arquivo
# gravado, valida de novo e devolve o SHA-256. **Esse hash identifica o arquivo que enviamos ao
# Canvas**, e reexecutar este notebook gera exatamente os mesmos bytes.
#
# Por fim, fazemos verificações de sanidade sem rótulos (elas não alteram o limiar nem as previsões):

# %%
# [TESTE]
p_te = positive_proba(final, X_te)
sanity = pd.DataFrame({
    "linhas": [len(X), len(X_te)],
    "% anomaly (rótulo / previsão)": [100 * pos.mean(), 100 * (y_te == config.POSITIVE).mean()],
    "% anomaly prevista no OOF": [100 * oof42["pred"].mean(), np.nan],
    "% com P entre 0,03 e 0,5": [100 * band.mean(), 100 * ((p_te >= 0.03) & (p_te < 0.5)).mean()],
}, index=["treino", "teste"])
display(sanity.round(2))
print("Previsões no teste:", pd.Series(y_te).value_counts().to_dict())

# %% [markdown]
# **Como interpretamos as verificações.** A proporção de `anomaly` prevista no teste (4,9%) está na
# mesma ordem da do treino (5,8% dos rótulos e 6,2% das previsões OOF), e a fração de linhas na faixa
# ambígua é igualmente pequena (0,16% no teste e 0,19% no treino). Isso confirma que o teste foi lido
# e processado corretamente. Essas verificações servem apenas para detectar problemas grosseiros: não
# usam rótulos e não alteram o arquivo, que já está gravado.
#
# ## 5. Explicabilidade do modelo final
#
# Para entender como o modelo decide, usamos quatro visões complementares, todas calculadas com o
# treino (explicar o modelo não exige o teste):
# 1. **Atributos que chegam ao modelo:** a confirmação de que o vazamento ficou de fora.
# 2. **Importância por permutação** nos atributos originais, em dados que o modelo não viu.
# 3. **SHAP** global (quais atributos mais pesam e em que direção) e local (por que uma conexão
#    específica foi ou não classificada como ataque).
# 4. **Árvore substituta:** regras legíveis que imitam o modelo, com a fidelidade medida.
#
# ### 5.1 Atributos que chegam ao modelo

# %%
model_features = list(final.estimator_[-1].feature_names_in_)
print(f"O HistGradientBoosting recebe {len(model_features)} atributos:")
print(", ".join(model_features))
forbidden = set(config.LEAKY_COLUMNS) | set(config.DERIVED_COLUMNS)
print("\nColunas de vazamento ou derivadas entre eles:", sorted(forbidden & set(model_features)) or "nenhuma")
print("Atributos de rede fora do modelo:", sorted(set(config.FEATURES) - set(model_features)))
assert not forbidden & set(model_features)

# %% [markdown]
# O modelo recebe os 41 atributos de rede, exceto `num_outbound_cmds`, que é constante (zero) em
# todo o treino. Confirmamos que nenhuma das 3 colunas de vazamento nem das 5 cópias derivadas chega
# ao modelo: todas as explicações a seguir se baseiam apenas em tráfego de rede.
#
# ### 5.2 Importância por permutação
#
# Embaralhar um atributo quebra a relação dele com a classe, e a queda de F1-macro mede quanto o
# modelo depende dele. Para medir a dependência real, e não a memorização, usamos um modelo treinado
# em 4 dos 5 folds (mesmo pipeline, mesmos hiperparâmetros) e calculamos a importância no fold que
# ficou de fora, com o limiar do modelo final. A permutação age nos **atributos originais**, antes da
# codificação.

# %%
cv5 = StratifiedGroupKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
tr, va = next(cv5.split(X, y, groups))
fold_model = build_final().fit(X.iloc[tr], y.iloc[tr])
X_va, y_va = X.iloc[va], y.iloc[va]


def f1_at_threshold(estimator, X_, y_):
    """F1-macro com o limiar do modelo final (o scorer padrão usaria 0,5)."""
    labels = np.where(positive_proba(estimator, X_) >= final.threshold_, config.POSITIVE, "normal")
    return f1_score(y_, labels, average="macro")


t0 = time.time()
perm = permutation_importance(fold_model, X_va, y_va, scoring=f1_at_threshold, n_repeats=5,
                              random_state=config.RANDOM_STATE, n_jobs=FOLD_JOBS)
perm = pd.DataFrame({"queda_f1_macro": perm.importances_mean, "dp": perm.importances_std},
                    index=X.columns).sort_values("queda_f1_macro", ascending=False)
print(f"F1-macro do fold de validação sem permutar: {f1_at_threshold(fold_model, X_va, y_va):.4f} "
      f"({len(va)} linhas, {time.time() - t0:.0f} s)")
display(perm.head(15).round(4))

fig, ax = plt.subplots(figsize=(8, 5))
top = perm.head(15).iloc[::-1]
ax.barh(top.index, top["queda_f1_macro"], xerr=top["dp"], color="#3b75af")
ax.set_xlabel("queda do F1-macro ao embaralhar o atributo")
ax.set_title("Importância por permutação (fold não visto)")
fig.tight_layout()
fig.savefig(config.FIGURES / "final_permutacao.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **O que observamos.**
# - **`service` é o atributo mais importante:** embaralhá-lo reduz o F1-macro em 0,035. Vários ataques
#   do NSL-KDD têm alvos característicos (`private`, `ecr_i`, `telnet`, `ftp_data`), então o serviço
#   já separa boa parte das classes.
# - Em seguida vêm os **volumes** (`dst_bytes`, `src_bytes`), as **taxas de erro por host**
#   (`dst_host_rerror_rate`) e atributos de conteúdo como `hot` (indicadores "quentes" na sessão,
#   sinal de ataques de acesso remoto).
# - As quedas individuais são moderadas porque vários atributos carregam informação parecida (por
#   exemplo, `flag` e as taxas de `serror`). O modelo tem redundância e, por isso, é robusto à perda de
#   um atributo isolado.
#
# ### 5.3 SHAP global
#
# Os valores SHAP repartem a saída do modelo para uma conexão (log-odds de `anomaly`) entre os
# atributos: valor positivo empurra para ataque, negativo para normal, e a soma dos valores com a
# média de referência reproduz exatamente a saída do modelo. Optamos pelo explicador por
# **permutação** do pacote `shap`, que só consulta o modelo e é exato para qualquer estimador. Antes,
# verificamos que o `TreeExplainer` não reproduz corretamente as divisões categóricas nativas do
# HistGradientBoosting (a soma dos valores não batia com a saída do modelo); com o explicador por
# permutação, a soma confere com a saída até a precisão numérica da máquina (teste em
# `tests/test_leakage.py`). Amostra: 1.000 linhas do treino, com 100 linhas de referência.

# %%
t0 = time.time()
sv, Xt_shap = shap_values(final, X, sample=1000, background=100)
f_final = anomaly_log_odds(final.estimator_[-1], list(Xt_shap.columns))
gap = np.abs(sv.values.sum(axis=1) + sv.base_values - f_final(Xt_shap.to_numpy())).max()
print(f"SHAP em {len(Xt_shap)} linhas ({time.time() - t0:.0f} s); maior diferença entre a soma "
      f"dos valores e a saída do modelo: {gap:.1e}")
shap_rank = pd.Series(np.abs(sv.values).mean(axis=0), index=Xt_shap.columns,
                      name="média |SHAP|").sort_values(ascending=False)

fig = plt.figure()
shap.plots.bar(sv, max_display=15, show=False)
plt.title("SHAP: importância média (log-odds de anomaly)")
plt.savefig(config.FIGURES / "final_shap_barras.png", dpi=150, bbox_inches="tight")
plt.show()
fig = plt.figure()
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("SHAP: direção do efeito de cada atributo")
plt.savefig(config.FIGURES / "final_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

compare = pd.DataFrame({"posição SHAP": range(1, 16)}, index=shap_rank.index[:15])
compare["posição permutação"] = [list(perm.index).index(c) + 1 for c in compare.index]
display(compare)

# %% [markdown]
# **O que observamos.**
# - **SHAP e permutação concordam no topo:** `service`, `dst_bytes` e `src_bytes` estão nas quatro
#   primeiras posições nas duas medidas. `flag` sobe no SHAP (4º contra 13º): ela influencia muito a
#   saída do modelo, mas sua informação se repete nas taxas de `serror`, então embaralhá-la sozinha
#   custa pouco F1.
# - **Direção dos efeitos** (beeswarm, cor = valor do atributo): poucos bytes de resposta
#   (`dst_bytes` baixo, azul) empurram para ataque, típico de varreduras e de conexões recusadas;
#   taxas de erro altas (`dst_host_rerror_rate`, `dst_host_srv_serror_rate`, vermelho) também
#   empurram para ataque, como em SYN flood e varredura de portas; `logged_in = 1` empurra para
#   normal. Para `service`, `flag` e `protocol_type`, a cor é o código ordinal da categoria e não tem
#   significado de "alto/baixo": vale a posição horizontal.
# - Os atributos de maior peso são os que a literatura de detecção de intrusão aponta (serviço, flag,
#   volumes e taxas de erro): o modelo aprendeu padrões reais de tráfego, não artefatos da base.
#
# ### 5.4 SHAP local: um ataque detectado e um caso difícil
#
# Escolhemos duas conexões do fold de validação da Seção 5.2, explicadas pelo modelo que **não** as
# viu no treino (o mesmo modelo de fold): um ataque detectado e um ataque que o modelo não sinalizou,
# ambos fora dos grupos contraditórios. Analisar os dois casos ajuda a entender tanto a lógica de
# detecção quanto o alcance dos atributos de tráfego.

# %%
p_va = positive_proba(fold_model, X_va)
pos_va, conf_va = pos[va], conflict[va]
hit = np.flatnonzero(pos_va & ~conf_va & (p_va >= final.threshold_))
miss = np.flatnonzero(pos_va & ~conf_va & (p_va < final.threshold_))
rng = np.random.default_rng(config.RANDOM_STATE)
chosen = {"acerto (ataque detectado)": rng.choice(hit), "erro (ataque não detectado)": rng.choice(miss)}
print(f"No fold: {len(hit)} ataques detectados e {len(miss)} não detectados fora dos conflitos")

bg = transformed(fold_model, X.iloc[tr].sample(100, random_state=config.RANDOM_STATE))
cols = list(bg.columns)
local_explainer = shap.Explainer(anomaly_log_odds(fold_model[-1], cols),
                                 shap.maskers.Independent(bg, max_samples=100),
                                 algorithm="permutation", seed=config.RANDOM_STATE)
show_cols = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "count", "srv_count",
             "hot", "num_failed_logins", "logged_in", "dst_host_count", "dst_host_srv_count"]
display(pd.DataFrame({name: X_va.iloc[i][show_cols] for name, i in chosen.items()}).T
        .assign(**{"P(anomaly)": [p_va[i] for i in chosen.values()]}))
for k, (name, i) in enumerate(chosen.items()):
    row = transformed(fold_model, X_va.iloc[[i]])
    explanation = local_explainer(row, max_evals=2 * len(cols) + 1, silent=True)[0]
    explanation.data = X_va.iloc[i][cols].to_numpy(dtype=object)  # categorias por nome no gráfico
    fig = plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f"{name}: P(anomaly) = {p_va[i]:.3f}")
    plt.savefig(config.FIGURES / f"final_shap_local_{k + 1}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# **O que observamos.**
# - **Ataque detectado:** conexão TCP ao serviço `daytime` com flag `S0` (pedido de abertura sem
#   resposta), sem bytes trocados, 276 conexões ao mesmo destino em 2 segundos e taxa de erro de SYN
#   igual a 1 no host. É o perfil de um SYN flood (negação de serviço): o modelo dá P(anomaly) = 1,000,
#   e todos os atributos principais (`flag = S0`, `dst_host_srv_serror_rate = 1`, o serviço e os bytes
#   zerados) empurram para ataque.
# - **Caso difícil:** sessão `telnet` com flag `SF` (conexão concluída normalmente), 2.376 bytes de
#   resposta e nenhum indicador de conteúdo (`hot` = 0, `num_failed_logins` = 0). O modelo reconhece o
#   serviço `telnet` como suspeito (+8,4 em log-odds), mas todo o resto se parece com uma sessão
#   comum, e a probabilidade final fica em 0,003. Ataques de acesso remoto que seguem o protocolo
#   normalmente são reconhecidos na literatura como difíceis de detectar só com atributos de tráfego:
#   o sinal estaria no conteúdo da sessão.
# - Casos assim são raros: no fold analisado, o modelo detectou 426 dos 441 ataques fora dos
#   conflitos (96,6%).
#
# ### 5.5 Árvore substituta
#
# Treinamos uma árvore de decisão rasa (profundidade 4) para imitar as **previsões** do modelo final
# no treino (não os rótulos). Ela usa os atributos originais, com os categóricos em one-hot, para que
# as regras sejam legíveis. A fidelidade é a fração de linhas em que a árvore concorda com o modelo.

# %%
rules, fidelity = surrogate_rules(final, X, max_depth=4)
agree = final.predict(X)
print(f"Fidelidade da árvore substituta (profundidade 4): {fidelity:.4f} "
      f"(prever sempre 'normal' concordaria em {np.mean(agree == 'normal'):.4f})")
print(rules)

# %% [markdown]
# **O que observamos.**
# - Com apenas 4 níveis, a árvore concorda com o modelo em **98,4%** das linhas (prever sempre
#   `normal` concordaria em 93,8%). As regras resumem bem a lógica do modelo:
#   - **`dst_host_srv_serror_rate` > 0,98 → `anomaly`** (salvo flags `S1`/`S2`, de conexões que
#     chegaram a ser estabelecidas): quase todas as conexões ao serviço daquele host terminaram em
#     erro de SYN, assinatura de SYN flood;
#   - **`same_srv_rate` ≤ 0,11 → `anomaly`** quando, fora da flag `SF`, `dst_host_diff_srv_rate`
#     > 0,04, ou, com `SF`, `dst_host_rerror_rate` > 0,16: poucas conexões ao mesmo serviço, muitos
#     serviços diferentes ou rejeições, assinatura de varredura de portas;
#   - **ICMP com `srv_count` > 12 → `anomaly`**: rajada de pings ao mesmo serviço (smurf, ipsweep);
#   - **flag `RSTR`** (conexão abortada pelo cliente) → `anomaly`.
# - A árvore é uma explicação aproximada: a previsão enviada é a do HistGradientBoosting.
#
# ## 6. Discussão dos resultados
#
# **Resultado esperado.** Estimamos o F1-macro no teste em **≈ 0,88** (CV agrupada 3×5: 0,8819 ±
# 0,0030), supondo que o teste tenha o mesmo ruído do treino. Se o teste tiver menos rótulos
# contraditórios, o valor tende a subir na direção de 0,98. Nosso modelo atinge mais de 97% do teto
# de 0,905 que os 1.702 grupos de rótulos contraditórios impõem a qualquer classificador.
#
# **Como obtivemos o número e por que confiamos nele.**
# 1. Validação cruzada **agrupada por duplicata**: as 13.067 cópias nunca ficam em lados opostos.
# 2. **Todo o pré-processamento dentro do pipeline**, ajustado só nos folds de treino, inclusive a
#    remoção de cópias e conflitos.
# 3. **Limiar cruzado:** cada fold é avaliado com um limiar escolhido nos outros folds.
# 4. **Métrica em todas as linhas**, exatamente como a oficial, com cópias e conflitos incluídos.
# 5. **Teste usado uma única vez**, depois do congelamento, apenas para `predict`.
#
# **A decisão de não usar as colunas de vazamento.** Na EDA (notebook 01), mostramos que um HGB com
# os 41 atributos de rede chega a F1-macro 0,875; com as três colunas de vazamento, a 0,891; e só com
# elas, sem nenhum dado de rede, a 0,889. Essas colunas são posteriores ao evento (triagem do SOC,
# tempo de resposta do analista) e se comportam como cópias ruidosas do rótulo: um detector que
# dependesse delas só "detectaria" o que o SOC já detectou. Abrimos mão de cerca de 0,015 no ranking
# em favor de um modelo metodologicamente correto e utilizável na prática.
#
# **O que mais contribuiu** (tabelas das Etapas 2 e 3, F1-macro de CV):
# - **Limpeza de rótulos** (`drop_inconsistent`) no boosting: +0,004 sobre não tratar a redundância
#   (0,8777 → 0,8821), o maior ganho de pré-processamento.
# - **Limiar ajustado por CV** em vez de 0,5: +0,002 a +0,003.
# - **Ausentes nativos** no HGB: mesmo desempenho da imputação pela mediana (0,8821 contra 0,8817),
#   com um passo a menos.
# - **Testamos e descartamos, com base na CV:** balanceamento, redução de dimensionalidade,
#   atributos derivados, ensembles e algoritmos fora da disciplina. Essas comparações mostraram que
#   o modelo já estava no patamar máximo: os 11 melhores candidatos ficaram dentro de 1
#   desvio-padrão, e escolhemos o mais simples e explicável.
#
# **Pontos de atenção.**
# - O rótulo verdadeiro das linhas contraditórias é desconhecido, e pode haver ruído de rótulo também
#   em linhas sem cópia, onde ele não é visível.
# - Cerca de 40% dos valores de `src_bytes` e de `dst_host_srv_count` estão ausentes; o modelo
#   aproveita a própria ausência como informação.
# - Ataques de acesso remoto que seguem o protocolo normalmente (Seção 5.4) têm pouco sinal nos
#   atributos de tráfego.
#
# **Trabalhos futuros** (ideias que **não** entraram no modelo submetido):
# - Calibração das probabilidades (Platt ou isotônica), para que P(anomaly) tenha leitura direta.
# - Modelos robustos a ruído de rótulo (perdas robustas, *co-teaching*), como alternativa a descartar
#   os conflitos.
# - Atributos de conteúdo mais ricos para ataques de acesso remoto, se a base os oferecesse.
#
# ## 7. Conclusões e checklist de entrega
#
# | Pergunta | Resposta |
# |---|---|
# | Modelo submetido | HistGradientBoosting (categorias e NaN nativos), `drop_inconsistent`, limiar 0,03 por CV |
# | **F1-macro esperado no teste** | **0,8819 ± 0,0030** (CV agrupada 3×5, todas as linhas) |
# | Métricas por classe esperadas | `normal` P 0,987 · R 0,984 · F1 0,986; `anomaly` P 0,759 · R 0,798 · F1 0,778 |
# | Acerto fora dos conflitos | 99,8% das conexões (F1-macro 0,983) |
# | Teto teórico da métrica | 0,905 (rótulos contraditórios); o modelo atinge mais de 97% dele |
# | Atributos mais importantes | `service`, `dst_bytes`, `src_bytes`, `flag`, taxas de erro por host |
# | Colunas de vazamento no modelo | nenhuma |
# | Reprodutibilidade | reexecutar o notebook gera o mesmo arquivo (mesmo SHA-256) |
#
# A célula abaixo confere os entregáveis desta etapa.

# %%
sub_path = config.SUBMISSION_FILE
registered = (config.RESULTS / f"submissao_{config.TEAM_NUMBER}.sha256").read_text(encoding="utf-8").split()[0]
print("Submissão:", sub_path.relative_to(config.ROOT).as_posix(), f"({sub_path.stat().st_size} bytes)")
print("SHA-256 do arquivo:   ", check_submission())
print("SHA-256 registrado:   ", registered)
print("Modelo final salvo:   ", (config.MODELS / "final.joblib").relative_to(config.ROOT).as_posix())
print("Figuras desta etapa:  ", ", ".join(sorted(p.name for p in config.FIGURES.glob("final_*.png"))))

# %% [markdown]
# **Checklist de envio (Canvas, até 28/11):**
# - [ ] Enviar `results/submissao_7.csv` sem abrir nem salvar em planilha (isso muda os bytes).
# - [ ] Conferir que o arquivo enviado tem o SHA-256 impresso acima (PowerShell:
#   `Get-FileHash results/submissao_7.csv`).
# - [ ] Entregar este notebook executado: reexecutá-lo do início ao fim gera o mesmo arquivo (mesmo
#   hash), como exige a Seção 3.1 do enunciado.
# - [ ] Incluir no relatório o F1-macro esperado (0,8819 ± 0,0030), a tabela por classe e a
#   declaração de uso de IA generativa (`reports/declaracao_ia.md`).
#
# **Auditoria de vazamento deste notebook.** Executamos
# `python -m nslkdd.audit notebooks/04_avaliacao_final.ipynb`: **0 erros** e 2 alertas, ambos
# esperados: (1) na Seção 3, `fit_resample` é chamado só para contar quantas linhas o
# `drop_inconsistent` mantém no ajuste, sem treinar nenhum modelo com essa saída; (2) na Seção 5.1,
# as colunas de vazamento são citadas apenas para confirmar que não chegam ao modelo. O teste é
# carregado uma única vez (Seção 4), depois do treino do modelo final.
