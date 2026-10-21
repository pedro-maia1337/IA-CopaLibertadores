"""Explicabilidade do modelo final (Etapa 4): importâncias, SHAP e regras substitutas."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config


def unwrap(model):
    """Devolve o Pipeline interno se o modelo estiver envolvido (ex.: WeightedDecisionClassifier)."""
    return getattr(model, "estimator_", model)


def transformed(pipe, X):
    """Aplica as etapas de transformação (samplers são ignorados no transform)."""
    pipe = unwrap(pipe)
    head = pipe[:-1]
    Xt = head.transform(X)
    try:
        names = head.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    return pd.DataFrame(Xt, columns=names, index=X.index)


def model_importances(pipe, X):
    """Importância nativa do estimador final (árvores/ensembles)."""
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


def shap_values(pipe, X, sample=2000):
    """Valores SHAP (TreeExplainer) do estimador final no espaço transformado."""
    import shap

    Xs = X.sample(min(sample, len(X)), random_state=config.RANDOM_STATE)
    Xt = transformed(pipe, Xs)
    final = unwrap(pipe)[-1]
    explainer = shap.TreeExplainer(getattr(final, "estimator_", final))  # EncodedLabelClassifier
    return explainer(Xt), Xt


def surrogate_rules(pipe, X, max_depth=4):
    """Árvore rasa que imita as previsões do modelo: regras legíveis + fidelidade."""
    Xt = transformed(pipe, X)
    y_model = pipe.predict(X)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=config.RANDOM_STATE)
    tree.fit(Xt, y_model)
    fidelity = float(np.mean(tree.predict(Xt) == y_model))
    return export_text(tree, feature_names=list(Xt.columns)), fidelity
