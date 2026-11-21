import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from nslkdd import audit, config, data
from nslkdd.decision import ThresholdClassifier, best_threshold, macro_f1_curve
from nslkdd.evaluation import ExperimentLog, binary_metrics, cross_validate_pipeline
from nslkdd.models import EncodedLabelClassifier, SoftVote
from nslkdd.preprocessing import DomainCleaner, _resolve_redundancy, build_pipeline


@pytest.fixture(scope="module")
def sample():
    X, y, groups = data.load_train()
    return X.iloc[:6000].reset_index(drop=True), y.iloc[:6000].reset_index(drop=True), groups[:6000]


def tree():
    return DecisionTreeClassifier(max_depth=6, random_state=0)


def test_samplers_only_touch_fit_data(sample):
    X, y, _ = sample
    pipe = build_pipeline(tree(), redundancy="drop_inconsistent", balancing="ros")
    pipe.fit(X, y)
    # predict não reamostra: uma previsão por linha recebida
    assert len(pipe.predict(X.iloc[:500])) == 500


@pytest.mark.parametrize("encoding", ["onehot", "onehot_rare", "frequency", "target", "ordinal"])
def test_every_service_encoding_builds(sample, encoding):
    X, y, _ = sample
    p = build_pipeline(tree(), service_encoding=encoding)
    assert np.isfinite(p.fit(X, y).predict_proba(X.iloc[:10])).all()


@pytest.mark.parametrize("imputation", ["mean", "median", "median_indicator", "iterative"])
def test_imputers_leave_no_missing(sample, imputation):
    X, y, _ = sample
    pipe = build_pipeline(tree(), imputation=imputation).fit(X, y)
    assert pipe[:-1].transform(X.iloc[:200]).notna().all().all()


def test_native_nan_keeps_missing_for_trees(sample):
    X, y, _ = sample
    pipe = build_pipeline(HistGradientBoostingClassifier(max_iter=10), imputation="none",
                          scaling="none").fit(X, y)
    assert pipe[:-1].transform(X).isna().any().any()


def test_resampling_without_imputation_is_refused():
    with pytest.raises(ValueError):
        build_pipeline(tree(), imputation="none", balancing="smote")


def test_domain_cleaner_removes_impossible_values():
    X, _, _ = data.load_train()
    limits = pd.Series(config.DOMAIN_MAX)
    assert (X[limits.index] > limits).any().any()
    for strategy in ("nan", "clip"):
        out = DomainCleaner(strategy).fit_transform(X)
        assert not (out[limits.index] > limits).any().any()
        assert not (out[config.NUMERIC] < 0).any().any()
    assert len(out) == len(X)


def test_domain_cleaner_does_not_create_missing_in_valid_rows():
    row = pd.DataFrame([{c: 0 for c in config.FEATURES}]).assign(
        protocol_type="tcp", service="http", flag="SF", same_srv_rate=1.0, count=511)
    out = DomainCleaner("nan").transform(row)
    assert out["same_srv_rate"].iat[0] == 1.0 and out["count"].iat[0] == 511


@pytest.mark.parametrize("strategy", ["drop_inconsistent", "majority_label", "relabel_model"])
def test_redundancy_leaves_no_conflict_or_copy(sample, strategy):
    X, y, _ = sample
    X2, y2 = _resolve_redundancy(X, y, strategy)
    g = data.duplicate_groups(X2)
    assert len(X2) == len(y2) and not pd.Series(g).duplicated().any()


def test_drop_duplicates_keeps_contradicting_pair(sample):
    X, y, _ = sample
    X2, y2 = _resolve_redundancy(X, y, "drop_duplicates")
    assert data.conflict_mask(data.duplicate_groups(X2), y2).sum() > 0


def test_cv_never_splits_copies(sample):
    X, y, groups = sample
    for train, val in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        assert not set(groups[train]) & set(groups[val])


def test_threshold_curve_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1
    proba = np.clip(y * 0.5 + rng.random(2000) * 0.6, 0, 1)
    t = np.array([0.1, 0.3, 0.5, 0.8])
    expected = [f1_score(y, proba >= v, average="macro") for v in t]
    assert np.allclose(macro_f1_curve(proba, y, t), expected)
    assert np.isclose(binary_metrics(y, proba >= 0.5)["f1_macro"], expected[2])
    assert 0 < best_threshold(proba, y) < 1


def test_threshold_classifier_uses_learned_cut(sample):
    X, y, _ = sample
    pipe = build_pipeline(tree(), redundancy="drop_inconsistent")
    model = ThresholdClassifier(pipe, n_splits=3, n_jobs=1).fit(X, y)
    assert 0 < model.threshold_ < 1
    assert set(model.predict(X.iloc[:50])) <= set(config.CLASSES)


def test_cv_reports_clean_and_noisy_metrics(sample):
    X, y, groups = sample
    pipe = build_pipeline(tree(), redundancy="drop_inconsistent")
    out = cross_validate_pipeline(pipe, X, y, groups, n_splits=3, n_jobs=1)
    assert out["f1_macro_mean"] > out["f1_macro_noisy_mean"]


def test_log_cache_depends_on_hyperparameters(sample, tmp_path):
    X, y, groups = sample
    log = ExperimentLog(tmp_path / "e.csv")
    log.run("t", "DT", tree(), X, y, groups, n_splits=3, n_jobs=1)
    log.run("t", "DT", tree(), X, y, groups, n_splits=3, n_jobs=1)
    assert len(log.table()) == 1
    other = DecisionTreeClassifier(max_depth=3, random_state=0)
    log.run("t", "DT", other, X, y, groups, n_splits=3, n_jobs=1)
    assert len(log.table()) == 2


def test_fitted_pipeline_can_be_saved(sample, tmp_path):
    X, y, _ = sample
    pipe = build_pipeline(RandomForestClassifier(n_estimators=5, random_state=0),
                          redundancy="drop_inconsistent", balancing="hybrid", min_count=600,
                          max_count=3000).fit(X, y)
    joblib.dump(pipe, tmp_path / "m.joblib")
    assert len(joblib.load(tmp_path / "m.joblib").predict(X.iloc[:5])) == 5


def test_xgboost_runs_with_native_missing(sample):
    from xgboost import XGBClassifier
    X, y, _ = sample
    model = EncodedLabelClassifier(XGBClassifier(n_estimators=10, tree_method="hist", n_jobs=1))
    pipe = build_pipeline(model, imputation="none", scaling="none").fit(X, y)
    assert set(pipe.predict(X.iloc[:20])) <= set(config.CLASSES)


def test_soft_vote_averages_members_with_different_pipelines(sample):
    X, y, _ = sample
    members = [("a", build_pipeline(tree())),
               ("b", build_pipeline(HistGradientBoostingClassifier(max_iter=10), imputation="none",
                                    scaling="none", redundancy="drop_inconsistent"))]
    vote = SoftVote(members).fit(X, y)
    expected = np.mean([m.predict_proba(X.iloc[:20]) for m in vote.members_], axis=0)
    assert np.allclose(vote.predict_proba(X.iloc[:20]), expected)
    assert ThresholdClassifier(SoftVote(members), n_splits=3, n_jobs=1).fit(X, y).threshold_ > 0


def test_leaky_columns_are_unreachable(sample):
    X, y, _ = sample
    frame = X.assign(**{c: 1.0 for c in config.LEAKY_COLUMNS})
    pipe = build_pipeline(tree()).fit(frame, y)
    names = set(pipe[:-1].transform(frame.iloc[:5]).columns)
    assert not names & set(config.LEAKY_COLUMNS)


def test_explanations_work_with_samplers_and_missing(sample):
    from nslkdd import explain
    from xgboost import XGBClassifier
    X, y, _ = sample
    model = EncodedLabelClassifier(XGBClassifier(n_estimators=10, tree_method="hist", n_jobs=1))
    pipe = build_pipeline(model, redundancy="drop_inconsistent", imputation="none",
                          scaling="none").fit(X, y)
    assert explain.model_importances(pipe, X).index[0] in set(explain.transformed(pipe, X).columns)
    rules, fidelity = explain.surrogate_rules(pipe, X.iloc[:500], max_depth=3)
    assert "class:" in rules and 0 <= fidelity <= 1
    values, Xt = explain.shap_values(pipe, X, sample=50)
    assert values.values.shape[:2] == Xt.shape
    wrapped = ThresholdClassifier(pipe, n_splits=3, n_jobs=1).fit(X, y)
    assert explain.transformed(wrapped, X.iloc[:5]).shape[0] == 5


def test_scanner_flags_bad_patterns(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": [
        "scaler.fit(X_test)\n", "X = pd.concat([train, test])\n", "pd.read_csv('KDDTest+.txt')\n",
        "X['score_ids_legado']\n"]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "02_bad.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    levels = [f["nivel"] for f in audit.scan_notebook(p)]
    assert levels.count("ERRO") == 3 and levels.count("ALERTA") == 1


def test_association_handles_missing():
    df = data.load_train_frame()
    out = audit.association_with_target(df[["src_bytes", "service", "tipo_alerta_soc"]].iloc[:5000],
                                        df[config.TARGET].iloc[:5000])
    assert out["info_mutua"].notna().all()


def test_feature_builder_is_row_wise_and_stateless(sample):
    from nslkdd.preprocessing import FeatureBuilder

    X, _, _ = sample
    fb = FeatureBuilder().fit(X.iloc[:100])
    full = fb.transform(X.iloc[:500])
    # o resultado de uma linha não depende das outras linhas nem do conjunto de ajuste
    alone = FeatureBuilder().fit(X.iloc[400:]).transform(X.iloc[[7]])
    pd.testing.assert_frame_equal(full.iloc[[7]], alone)
    assert len(full) == 500 and list(full.columns[-len(FeatureBuilder.NAMES):]) == FeatureBuilder.NAMES
    assert not np.isinf(full[FeatureBuilder.NAMES].to_numpy(dtype=float)).any()


def test_engineered_pipeline_uses_only_network_features(sample):
    from nslkdd.models import make_model

    X, y, _ = sample
    pipe = build_pipeline(make_model("HGB_native", max_iter=20), redundancy="majority_label",
                          imputation="none", scaling="none", log_transform=False,
                          service_encoding="ordinal", categorical_encoding="ordinal",
                          engineered=True).fit(X, y)
    assert not set(config.LEAKY_COLUMNS) & set(pipe[:-1].get_feature_names_out())
    assert len(pipe.predict(X.iloc[:300])) == 300
