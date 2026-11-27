"""Explicabilidade do modelo final (Etapa 4): importâncias, SHAP e regras substitutas."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config


def unwrap(model):
    """Devolve o Pipeline interno se o modelo estiver envolvido (ex.: ThresholdClassifier)."""
    return getattr(model, "estimator_", model)


def transformed(pipe, X):
    """Aplica as etapas de transformação (samplers são ignorados no transform).

    O pré-processador devolve DataFrame, então os nomes das colunas já vêm prontos. Depois
    de PCA ou seleção de atributos, os nomes são os gerados pelo passo de redução.
    """
    Xt = unwrap(pipe)[:-1].transform(X)
    return pd.DataFrame(np.asarray(Xt, dtype=float), columns=getattr(Xt, "columns", None),
                        index=X.index)


def model_importances(pipe, X):
    """Importância nativa do estimador final (árvores/ensembles com feature_importances_)."""
    model = unwrap(pipe)[-1]
    names = transformed(pipe, X.iloc[:5]).columns
    return (pd.Series(model.feature_importances_, index=names)
            .sort_values(ascending=False).rename("importancia"))


def permutation_importances(pipe, X, y, n_repeats=5, scoring="f1_macro", sample=20000):
    """Importância por permutação sobre os atributos ORIGINAIS (antes da codificação)."""
    if len(X) > sample:
        X = X.sample(sample, random_state=config.RANDOM_STATE)
        y = y.loc[X.index]
    r = permutation_importance(pipe, X, y, n_repeats=n_repeats, scoring=scoring,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    return pd.DataFrame({"media": r.importances_mean, "desvio": r.importances_std},
                        index=X.columns).sort_values("media", ascending=False)


def anomaly_log_odds(final, columns):
    """Função A -> log-odds de 'anomaly' do estimador final, no espaço transformado."""
    def f(A):
        Xa = pd.DataFrame(A, columns=columns)
        if hasattr(final, "decision_function") and len(final.classes_) == 2:
            d = final.decision_function(Xa)  # log-odds da classe classes_[1]
            return d if final.classes_[1] == config.POSITIVE else -d
        p = final.predict_proba(Xa)[:, list(final.classes_).index(config.POSITIVE)]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))
    return f


def shap_values(pipe, X, sample=1000, background=100):
    """Valores SHAP do estimador final, em log-odds de 'anomaly', no espaço transformado.

    Usa o explicador por permutação, que só chama o modelo: é exato (aditivo) para qualquer
    estimador. O TreeExplainer não serve aqui, porque não reproduz as divisões categóricas
    nativas do HistGradientBoosting (a soma dos valores não bate com a saída do modelo).
    O fundo (`background`) e as linhas explicadas são amostras de X.
    """
    import shap

    Xs = X.sample(min(sample, len(X)), random_state=config.RANDOM_STATE)
    Xb = X.sample(min(background, len(X)), random_state=config.RANDOM_STATE + 1)
    Xt, Xbt = transformed(pipe, Xs), transformed(pipe, Xb)
    f = anomaly_log_odds(unwrap(pipe)[-1], list(Xt.columns))
    explainer = shap.Explainer(f, shap.maskers.Independent(Xbt, max_samples=len(Xbt)),
                               algorithm="permutation", seed=config.RANDOM_STATE)
    return explainer(Xt, max_evals=2 * Xt.shape[1] + 1, silent=True), Xt


def readable_frame(pipe, X):
    """Atributos originais (após a limpeza de domínio do pipeline) com categóricos em one-hot.

    Serve só para a árvore substituta: regras como `flag_SF <= 0.5` são legíveis, ao contrário
    de cortes sobre códigos ordinais. Não entra no modelo.
    """
    inner = unwrap(pipe)
    if "domain" in inner.named_steps:
        X = inner.named_steps["domain"].transform(X)
    return pd.get_dummies(X, columns=config.CATEGORICAL, dtype=float)


def surrogate_rules(pipe, X, max_depth=4, readable=True):
    """Árvore rasa que imita as previsões do modelo: regras legíveis + fidelidade.

    readable=True: a árvore usa os atributos originais com categóricos em one-hot
    (`readable_frame`); False: o espaço transformado do pipeline. A árvore do scikit-learn
    aceita NaN, então funciona também com imputação 'none'.
    """
    Xt = readable_frame(pipe, X) if readable else transformed(pipe, X)
    y_model = pipe.predict(X)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=config.RANDOM_STATE)
    tree.fit(Xt, y_model)
    fidelity = float(np.mean(tree.predict(Xt) == y_model))
    return export_text(tree, feature_names=list(Xt.columns)), fidelity
